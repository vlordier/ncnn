// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_convert_loftrfinematchpostprocess.h"

#include <algorithm>
#include <cstdlib>

namespace pnnx {

namespace ncnn {

static bool is_type(const Operator* op, const char* type)
{
    return op && op->type == type;
}

static Operator* producer(Operand* r)
{
    return r ? r->producer : 0;
}

static bool collect_branch_inputs(Operand* branch_points, Operand* branch_conf,
                                  Operand*& mkpts, Operand*& mconf,
                                  Operand*& m_bids,
                                  int& topk_k,
                                  int& batch_value)
{
    // branch_points := torch.where(valid.unsqueeze(-1), gathered_points, zeros)
    Operator* where_points = producer(branch_points);
    if (!is_type(where_points, "torch.where") || where_points->inputs.size() != 3)
        return false;

    Operator* gather_points = producer(where_points->inputs[1]);
    if (!is_type(gather_points, "Tensor.index") || gather_points->inputs.size() < 1)
        return false;

    Operator* cat_points = producer(gather_points->inputs[0]);
    if (!is_type(cat_points, "torch.cat") || cat_points->inputs.size() != 2)
        return false;

    Operand* mkpts_candidate = cat_points->inputs[0];

    // branch_conf := torch.where(valid, topk_scores, minus_one)
    Operator* where_conf = producer(branch_conf);
    if (!is_type(where_conf, "torch.where") || where_conf->inputs.size() != 3)
        return false;

    Operand* valid = where_conf->inputs[0];
    Operand* topk_scores = where_conf->inputs[1];

    Operator* topk = producer(topk_scores);
    if (!is_type(topk, "torch.topk") || topk->inputs.size() != 1)
        return false;

    if (!topk->has_param("k"))
        return false;

    const Parameter& k = topk->params.at("k");
    if (k.type != 2)
        return false;

    const int k_val = k.i;

    Operator* cat_scores = producer(topk->inputs[0]);
    if (!is_type(cat_scores, "torch.cat") || cat_scores->inputs.size() != 2)
        return false;

    Operator* where_masked_scores = producer(cat_scores->inputs[0]);
    if (!is_type(where_masked_scores, "torch.where") || where_masked_scores->inputs.size() < 2)
        return false;

    Operand* mconf_candidate = where_masked_scores->inputs[1];

    Operator* eq = producer(where_masked_scores->inputs[0]);
    if (!is_type(eq, "torch.eq") || eq->inputs.size() < 1)
        return false;

    Operand* m_bids_candidate = eq->inputs[0];

    if (!eq->has_param("other"))
        return false;

    const Parameter& other = eq->params.at("other");
    int bid_val = 0;
    if (other.type == 2)
    {
        bid_val = other.i;
    }
    else if (other.type == 3)
    {
        bid_val = (int)(other.f + (other.f >= 0.f ? 0.5f : -0.5f));
    }
    else
    {
        return false;
    }

    // valid mask semantics are produced by prior ops, but may already be lowered
    // from pnnx.Expression into primitive ops by expand_expression.
    if (!valid)
        return false;

    mkpts = mkpts_candidate;
    mconf = mconf_candidate;
    m_bids = m_bids_candidate;
    topk_k = k_val;
    batch_value = bid_val;

    return true;
}

void fuse_convert_loftrfinematchpostprocess(Graph& graph)
{
    int fused_index = 0;
    const bool debug = (std::getenv("PNNX_DEBUG_LOFTR_FUSE") != 0);

    for (Operator* op : graph.ops)
    {
        if (op->type != "pnnx.Output")
            continue;

        if (debug)
        {
            fprintf(stderr, "[loftr_fuse] inspect output %s inputs=%d\n", op->name.c_str(), (int)op->inputs.size());
        }

        Operator* stack0 = 0;
        Operator* stack1 = 0;
        Operator* stack2 = 0;

        if (op->inputs.size() == 3)
        {
            // chain_multi_output flattened tuple output
            stack0 = producer(op->inputs[0]);
            stack1 = producer(op->inputs[1]);
            stack2 = producer(op->inputs[2]);
        }
        else if (op->inputs.size() == 1)
        {
            // fallback if tuple output is still present
            Operator* tuple = producer(op->inputs[0]);
            if (!is_type(tuple, "prim::TupleConstruct") || tuple->inputs.size() != 3)
                continue;

            stack0 = producer(tuple->inputs[0]);
            stack1 = producer(tuple->inputs[1]);
            stack2 = producer(tuple->inputs[2]);
        }
        else
        {
            continue;
        }

        if (!is_type(stack0, "torch.stack") || !is_type(stack1, "torch.stack") || !is_type(stack2, "torch.stack"))
        {
            if (debug)
            {
                fprintf(stderr, "[loftr_fuse] skip output %s not stack trio\n", op->name.c_str());
            }
            continue;
        }
        if (stack0->inputs.size() != 2 || stack1->inputs.size() != 2 || stack2->inputs.size() != 2)
            continue;

        Operand* mkpts0_a = 0;
        Operand* mconf_a = 0;
        Operand* m_bids_a = 0;
        int topk_a = 0;
        int bid_a0 = -1;

        Operand* mkpts1_a = 0;
        Operand* mconf_a2 = 0;
        Operand* m_bids_a2 = 0;
        int topk_a2 = 0;
        int bid_a1 = -1;

        Operand* mkpts0_b = 0;
        Operand* mconf_b = 0;
        Operand* m_bids_b = 0;
        int topk_b = 0;
        int bid_b0 = -1;

        Operand* mkpts1_b = 0;
        Operand* mconf_b2 = 0;
        Operand* m_bids_b2 = 0;
        int topk_b2 = 0;
        int bid_b1 = -1;

        if (!collect_branch_inputs(stack0->inputs[0], stack2->inputs[0], mkpts0_a, mconf_a, m_bids_a, topk_a, bid_a0))
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch A0 collect failed\n");
            continue;
        }
        if (!collect_branch_inputs(stack1->inputs[0], stack2->inputs[0], mkpts1_a, mconf_a2, m_bids_a2, topk_a2, bid_a1))
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch A1 collect failed\n");
            continue;
        }

        if (!collect_branch_inputs(stack0->inputs[1], stack2->inputs[1], mkpts0_b, mconf_b, m_bids_b, topk_b, bid_b0))
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch B0 collect failed\n");
            continue;
        }
        if (!collect_branch_inputs(stack1->inputs[1], stack2->inputs[1], mkpts1_b, mconf_b2, m_bids_b2, topk_b2, bid_b1))
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch B1 collect failed\n");
            continue;
        }

        if (mkpts0_a != mkpts0_b || mkpts1_a != mkpts1_b)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] mkpts candidate mismatch\n");
            continue;
        }
        if (mconf_a != mconf_a2)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch A mconf mismatch\n");
            continue;
        }
        if (mconf_b != mconf_b2)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] branch B mconf mismatch\n");
            continue;
        }
        if (topk_a != topk_a2 || topk_a != topk_b || topk_a != topk_b2)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] topk mismatch\n");
            continue;
        }
        if (mconf_a != mconf_a2 || mconf_a != mconf_b || mconf_a != mconf_b2)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] mconf mismatch\n");
            continue;
        }
        if (bid_a0 != bid_a1 || bid_b0 != bid_b1 || bid_a0 == bid_b0)
        {
            if (debug) fprintf(stderr, "[loftr_fuse] batch id pattern mismatch\n");
            continue;
        }

        if (debug)
        {
            fprintf(stderr, "[loftr_fuse] matched output %s topk=%d\n", op->name.c_str(), topk_a);
        }
        Operator* fused = graph.new_operator_before("LoFTRFineMatchPostprocess", "loftr_finematchpostprocess_" + std::to_string(fused_index++), op);
        fused->params["0"] = topk_a;
        const int expected_batch = std::max(bid_a0, bid_b0) + 1;
        fused->params["1"] = expected_batch;
        fused->inputnames = std::vector<std::string> {"mkpts0", "mkpts1", "mconf", "m_bids"};
        fused->inputs = std::vector<Operand*> {mkpts0_a, mkpts1_a, mconf_a, m_bids_a};
        mkpts0_a->consumers.push_back(fused);
        mkpts1_a->consumers.push_back(fused);
        mconf_a->consumers.push_back(fused);
        m_bids_a->consumers.push_back(fused);

        Operand* stack_outs[3] = {stack0->outputs.empty() ? 0 : stack0->outputs[0],
                                  stack1->outputs.empty() ? 0 : stack1->outputs[0],
                                  stack2->outputs.empty() ? 0 : stack2->outputs[0]
                                 };
        if (!stack_outs[0] || !stack_outs[1] || !stack_outs[2])
        {
            if (debug) fprintf(stderr, "[loftr_fuse] stack output missing\n");
            continue;
        }

        for (int i = 0; i < 3; i++)
        {
            Operand* out = stack_outs[i];
            Operator* old_producer = out->producer;
            if (old_producer)
                old_producer->outputs.erase(std::remove(old_producer->outputs.begin(), old_producer->outputs.end(), out), old_producer->outputs.end());

            // Keep compatibility with downstream ncnn reshape fixup that expects
            // the pre-reshape blob name emitted by original torch.stack lowering.
            const std::string pre_reshape_name = (i == 0 ? stack0->name : (i == 1 ? stack1->name : stack2->name)) + "_ncnnreshape_in";
            out->name = pre_reshape_name;

            out->producer = fused;
            fused->outputs.push_back(out);
        }

        // Mark original stack ops as already-lowered so convert_torch_stack
        // will not inject reshape placeholders for now-disconnected branches.
        stack0->type = "Concat";
        stack1->type = "Concat";
        stack2->type = "Concat";
    }

    if (debug)
    {
        fprintf(stderr, "[loftr_fuse] fused_count=%d\n", fused_index);
    }
}

} // namespace ncnn

} // namespace pnnx

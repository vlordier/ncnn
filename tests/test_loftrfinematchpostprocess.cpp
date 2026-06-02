// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer.h"

#include <vector>

static int run_layer(const std::vector<ncnn::Mat>& bottoms, std::vector<ncnn::Mat>& tops, int topk, int bidirectional)
{
    ncnn::Layer* op = ncnn::create_layer("LoFTRFineMatchPostprocess");
    if (!op)
    {
        fprintf(stderr, "create_layer LoFTRFineMatchPostprocess failed\n");
        return -1;
    }

    ncnn::ParamDict pd;
    pd.set(0, topk);
    pd.set(1, bidirectional);

    ncnn::Option opt;
    opt.num_threads = 1;

    int ret = op->load_param(pd);
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret == 0)
        ret = op->forward(bottoms, tops, opt);

    op->destroy_pipeline(opt);
    delete op;

    return ret;
}

static int test_flat_with_batch_ids()
{
    const int N = 5;
    ncnn::Mat mkpts0(2, N);
    ncnn::Mat mkpts1(2, N);
    ncnn::Mat mconf(N);
    ncnn::Mat mbids(N);

    // idx0
    mkpts0.row(0)[0] = 10.f;
    mkpts0.row(0)[1] = 0.f;
    mkpts1.row(0)[0] = 100.f;
    mkpts1.row(0)[1] = 0.f;
    mconf[0] = 0.2f;
    mbids[0] = 0.f;

    // idx1
    mkpts0.row(1)[0] = 11.f;
    mkpts0.row(1)[1] = 1.f;
    mkpts1.row(1)[0] = 101.f;
    mkpts1.row(1)[1] = 1.f;
    mconf[1] = 0.9f;
    mbids[1] = 1.f;

    // idx2 invalid confidence
    mkpts0.row(2)[0] = 12.f;
    mkpts0.row(2)[1] = 2.f;
    mkpts1.row(2)[0] = 102.f;
    mkpts1.row(2)[1] = 2.f;
    mconf[2] = -1e9f;
    mbids[2] = 0.f;

    // idx3
    mkpts0.row(3)[0] = 13.f;
    mkpts0.row(3)[1] = 3.f;
    mkpts1.row(3)[0] = 103.f;
    mkpts1.row(3)[1] = 3.f;
    mconf[3] = 0.7f;
    mbids[3] = 1.f;

    // idx4
    mkpts0.row(4)[0] = 14.f;
    mkpts0.row(4)[1] = 4.f;
    mkpts1.row(4)[0] = 104.f;
    mkpts1.row(4)[1] = 4.f;
    mconf[4] = 0.5f;
    mbids[4] = 0.f;

    std::vector<ncnn::Mat> bottoms(4);
    bottoms[0] = mkpts0;
    bottoms[1] = mkpts1;
    bottoms[2] = mconf;
    bottoms[3] = mbids;

    std::vector<ncnn::Mat> tops;
    int ret = run_layer(bottoms, tops, 3, 0);
    if (ret != 0)
    {
        fprintf(stderr, "run_layer flat_with_batch_ids failed ret=%d\n", ret);
        return ret;
    }

    ncnn::Mat expected0(2, 3, 2);
    ncnn::Mat expected1(2, 3, 2);
    ncnn::Mat expectedc(3, 2);
    expected0.fill(0.f);
    expected1.fill(0.f);
    expectedc.fill(-1.f);

    // batch 0: idx4 (0.5), idx0 (0.2), pad
    expected0.channel(0).row(0)[0] = 14.f;
    expected0.channel(0).row(0)[1] = 4.f;
    expected1.channel(0).row(0)[0] = 104.f;
    expected1.channel(0).row(0)[1] = 4.f;
    expectedc.row(0)[0] = 0.5f;

    expected0.channel(0).row(1)[0] = 10.f;
    expected0.channel(0).row(1)[1] = 0.f;
    expected1.channel(0).row(1)[0] = 100.f;
    expected1.channel(0).row(1)[1] = 0.f;
    expectedc.row(0)[1] = 0.2f;

    // batch 1: idx1 (0.9), idx3 (0.7), pad
    expected0.channel(1).row(0)[0] = 11.f;
    expected0.channel(1).row(0)[1] = 1.f;
    expected1.channel(1).row(0)[0] = 101.f;
    expected1.channel(1).row(0)[1] = 1.f;
    expectedc.row(1)[0] = 0.9f;

    expected0.channel(1).row(1)[0] = 13.f;
    expected0.channel(1).row(1)[1] = 3.f;
    expected1.channel(1).row(1)[0] = 103.f;
    expected1.channel(1).row(1)[1] = 3.f;
    expectedc.row(1)[1] = 0.7f;

    if (Compare(tops[0], expected0, 0.0001f) != 0 || Compare(tops[1], expected1, 0.0001f) != 0 || Compare(tops[2], expectedc, 0.0001f) != 0)
    {
        fprintf(stderr, "test_flat_with_batch_ids output mismatch\n");
        return -1;
    }

    return 0;
}

static int test_prebatched()
{
    ncnn::Mat mkpts0(2, 4, 2);
    ncnn::Mat mkpts1(2, 4, 2);
    ncnn::Mat mconf(4, 2);

    // batch 0
    mkpts0.channel(0).row(0)[0] = 1.f;
    mkpts0.channel(0).row(0)[1] = 10.f;
    mkpts1.channel(0).row(0)[0] = 2.f;
    mkpts1.channel(0).row(0)[1] = 20.f;
    mconf.row(0)[0] = 0.1f;

    mkpts0.channel(0).row(1)[0] = 3.f;
    mkpts0.channel(0).row(1)[1] = 30.f;
    mkpts1.channel(0).row(1)[0] = 4.f;
    mkpts1.channel(0).row(1)[1] = 40.f;
    mconf.row(0)[1] = 0.8f;

    mkpts0.channel(0).row(2)[0] = 5.f;
    mkpts0.channel(0).row(2)[1] = 50.f;
    mkpts1.channel(0).row(2)[0] = 6.f;
    mkpts1.channel(0).row(2)[1] = 60.f;
    mconf.row(0)[2] = -1e9f;

    mkpts0.channel(0).row(3)[0] = 7.f;
    mkpts0.channel(0).row(3)[1] = 70.f;
    mkpts1.channel(0).row(3)[0] = 8.f;
    mkpts1.channel(0).row(3)[1] = 80.f;
    mconf.row(0)[3] = 0.5f;

    // batch 1
    mkpts0.channel(1).row(0)[0] = 11.f;
    mkpts0.channel(1).row(0)[1] = 110.f;
    mkpts1.channel(1).row(0)[0] = 12.f;
    mkpts1.channel(1).row(0)[1] = 120.f;
    mconf.row(1)[0] = 0.9f;

    mkpts0.channel(1).row(1)[0] = 13.f;
    mkpts0.channel(1).row(1)[1] = 130.f;
    mkpts1.channel(1).row(1)[0] = 14.f;
    mkpts1.channel(1).row(1)[1] = 140.f;
    mconf.row(1)[1] = 0.2f;

    mkpts0.channel(1).row(2)[0] = 15.f;
    mkpts0.channel(1).row(2)[1] = 150.f;
    mkpts1.channel(1).row(2)[0] = 16.f;
    mkpts1.channel(1).row(2)[1] = 160.f;
    mconf.row(1)[2] = 0.6f;

    mkpts0.channel(1).row(3)[0] = 17.f;
    mkpts0.channel(1).row(3)[1] = 170.f;
    mkpts1.channel(1).row(3)[0] = 18.f;
    mkpts1.channel(1).row(3)[1] = 180.f;
    mconf.row(1)[3] = -1e9f;

    std::vector<ncnn::Mat> bottoms(3);
    bottoms[0] = mkpts0;
    bottoms[1] = mkpts1;
    bottoms[2] = mconf;

    std::vector<ncnn::Mat> tops;
    int ret = run_layer(bottoms, tops, 2, 1);
    if (ret != 0)
    {
        fprintf(stderr, "run_layer prebatched failed ret=%d\n", ret);
        return ret;
    }

    ncnn::Mat expected0(2, 2, 2);
    ncnn::Mat expected1(2, 2, 2);
    ncnn::Mat expectedc(2, 2);
    expected0.fill(0.f);
    expected1.fill(0.f);
    expectedc.fill(-1.f);

    // batch 0: idx1 (0.8), idx3 (0.5)
    expected0.channel(0).row(0)[0] = 3.f;
    expected0.channel(0).row(0)[1] = 30.f;
    expected1.channel(0).row(0)[0] = 4.f;
    expected1.channel(0).row(0)[1] = 40.f;
    expectedc.row(0)[0] = 0.8f;

    expected0.channel(0).row(1)[0] = 7.f;
    expected0.channel(0).row(1)[1] = 70.f;
    expected1.channel(0).row(1)[0] = 8.f;
    expected1.channel(0).row(1)[1] = 80.f;
    expectedc.row(0)[1] = 0.5f;

    // batch 1: idx0 (0.9), idx2 (0.6)
    expected0.channel(1).row(0)[0] = 11.f;
    expected0.channel(1).row(0)[1] = 110.f;
    expected1.channel(1).row(0)[0] = 12.f;
    expected1.channel(1).row(0)[1] = 120.f;
    expectedc.row(1)[0] = 0.9f;

    expected0.channel(1).row(1)[0] = 15.f;
    expected0.channel(1).row(1)[1] = 150.f;
    expected1.channel(1).row(1)[0] = 16.f;
    expected1.channel(1).row(1)[1] = 160.f;
    expectedc.row(1)[1] = 0.6f;

    if (Compare(tops[0], expected0, 0.0001f) != 0 || Compare(tops[1], expected1, 0.0001f) != 0 || Compare(tops[2], expectedc, 0.0001f) != 0)
    {
        fprintf(stderr, "test_prebatched output mismatch\n");
        return -1;
    }

    return 0;
}

int main()
{
    return 0
           || test_flat_with_batch_ids()
           || test_prebatched();
}

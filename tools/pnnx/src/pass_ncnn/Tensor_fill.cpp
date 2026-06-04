// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_fill : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.fill             op_0        1 1 input out value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "BinaryOp";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("value").type == 2)
            return captured_params.at("value").i == 0;
        if (captured_params.at("value").type == 3)
            return captured_params.at("value").f == 0.f;

        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // Tensor.fill is only expected as value=None/0 in this LoFTR path.
        // Lower to x * 0 to avoid introducing a dedicated runtime layer.
        op->params["0"] = 2;
        op->params["1"] = 1;
        op->params["2"] = 0.f;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_fill, 20)

class Tensor_fill_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.fill             op_0        1 1 input out value=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "BinaryOp";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 2;
        op->params["1"] = 1;
        op->params["2"] = 0.f;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_fill_1, 20)

} // namespace ncnn

} // namespace pnnx

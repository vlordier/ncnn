// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_COMPARE_H
#define LAYER_COMPARE_H

#include "layer.h"

namespace ncnn {

class Compare : public Layer
{
public:
    Compare();

    virtual int load_param(const ParamDict& pd);

    using Layer::forward;
    using Layer::forward_inplace;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum OperationType
    {
        Operation_EQ = 0,
        Operation_NE = 1,
        Operation_GT = 2,
        Operation_GE = 3,
        Operation_LT = 4,
        Operation_LE = 5
    };

public:
    int op_type;
    int with_scalar;
    float b;
};

} // namespace ncnn

#endif // LAYER_COMPARE_H

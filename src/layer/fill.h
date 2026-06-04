// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FILL_H
#define LAYER_FILL_H

#include "layer.h"

namespace ncnn {

class Fill : public Layer
{
public:
    Fill();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    float value;
};

} // namespace ncnn

#endif // LAYER_FILL_H

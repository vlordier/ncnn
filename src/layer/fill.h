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

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    float value;
};

} // namespace ncnn

#endif // LAYER_FILL_H

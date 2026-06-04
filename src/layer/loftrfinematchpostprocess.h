// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LOFTRFINEMATCHPOSTPROCESS_H
#define LAYER_LOFTRFINEMATCHPOSTPROCESS_H

#include "layer.h"

namespace ncnn {

class LoFTRFineMatchPostprocess : public Layer
{
public:
    LoFTRFineMatchPostprocess();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // Placeholder parameters for the upcoming fused implementation.
    int topk;
    int bidirectional;
};

} // namespace ncnn

#endif // LAYER_LOFTRFINEMATCHPOSTPROCESS_H

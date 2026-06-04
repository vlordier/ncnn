// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fill.h"

namespace ncnn {

Fill::Fill()
{
    one_blob_only = true;
    support_inplace = false;

    value = 0.f;
}

int Fill::load_param(const ParamDict& pd)
{
    value = pd.get(0, 0.f);
    return 0;
}

int Fill::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    top_blob.fill(value);

    return 0;
}

} // namespace ncnn

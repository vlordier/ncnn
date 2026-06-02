// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "loftrfinematchpostprocess.h"

namespace ncnn {

LoFTRFineMatchPostprocess::LoFTRFineMatchPostprocess()
{
    one_blob_only = false;
    support_inplace = false;

    topk = 512;
    bidirectional = 1;
}

int LoFTRFineMatchPostprocess::load_param(const ParamDict& pd)
{
    topk = pd.get(0, 512);
    bidirectional = pd.get(1, 1);

    return 0;
}

int LoFTRFineMatchPostprocess::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const
{
    // PR1 scaffold behavior:
    // - keep interface and registration stable
    // - return three outputs so graph wiring can be tested immediately
    // - fused LoFTR fine-match logic lands in follow-up commits
    if (bottom_blobs.size() < 3)
        return -1;

    top_blobs.resize(3);
    top_blobs[0] = bottom_blobs[0];
    top_blobs[1] = bottom_blobs[1];
    top_blobs[2] = bottom_blobs[2];

    return 0;
}

} // namespace ncnn

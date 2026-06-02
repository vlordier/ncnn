// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_loftrfinematchpostprocess(const ncnn::Mat& mkpts0, const ncnn::Mat& mkpts1, const ncnn::Mat& mconf, int topk, int bidirectional)
{
    ncnn::ParamDict pd;
    pd.set(0, topk);
    pd.set(1, bidirectional);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = mkpts0;
    as[1] = mkpts1;
    as[2] = mconf;

    int ret = test_layer("LoFTRFineMatchPostprocess", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_loftrfinematchpostprocess failed topk=%d bidirectional=%d\n", topk, bidirectional);
    }

    return ret;
}

int main()
{
    SRAND(776757);

    ncnn::Mat mkpts0 = RandomMat(2, 512, 2);
    ncnn::Mat mkpts1 = RandomMat(2, 512, 2);
    ncnn::Mat mconf = RandomMat(512, 2);

    return 0
           || test_loftrfinematchpostprocess(mkpts0, mkpts1, mconf, 512, 1)
           || test_loftrfinematchpostprocess(mkpts0, mkpts1, mconf, 256, 0);
}

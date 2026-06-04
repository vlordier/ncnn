// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "loftrfinematchpostprocess.h"

#include <algorithm>
#include <stdint.h>
#include <vector>

namespace ncnn {

LoFTRFineMatchPostprocess::LoFTRFineMatchPostprocess()
{
    one_blob_only = false;
    support_inplace = false;

    topk = 512;
    expected_branches = 1;
}

int LoFTRFineMatchPostprocess::load_param(const ParamDict& pd)
{
    topk = pd.get(0, 512);
    expected_branches = pd.get(1, 1);

    return 0;
}

int LoFTRFineMatchPostprocess::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const
{
    if (bottom_blobs.size() < 3)
        return -1;

    const Mat& mkpts0 = bottom_blobs[0];
    const Mat& mkpts1 = bottom_blobs[1];
    const Mat& mconf = bottom_blobs[2];

    if (mkpts0.dims != mkpts1.dims || mkpts0.w != mkpts1.w || mkpts0.h != mkpts1.h || mkpts0.c != mkpts1.c)
        return -1;
    if (mkpts0.w != 2)
        return -1;

    const bool has_m_bids = bottom_blobs.size() >= 4;
    const Mat* m_bids = has_m_bids ? &bottom_blobs[3] : 0;

    auto get_m_bid = [&](int i) -> int {
        if (!m_bids)
            return 0;

        if (m_bids->elemsize == 8u)
        {
            const int64_t* p = (const int64_t*)m_bids->data;
            return (int)p[i];
        }

        if (m_bids->elemsize == 4u)
        {
            const float* p = (const float*)m_bids->data;
            const float bidf = p[i];
            return (int)(bidf + (bidf >= 0.f ? 0.5f : -0.5f));
        }

        const float bidf = m_bids->operator[](i);
        return (int)(bidf + (bidf >= 0.f ? 0.5f : -0.5f));
    };

    const int K = topk > 0 ? topk : 1;
    const float neg_inf_half = -5e8f;

    int B = 1;
    int N = 0;
    bool prebatched = false;

    if (mkpts0.dims == 3)
    {
        // [B, N, 2] laid out as ncnn [w=2, h=N, c=B]
        prebatched = true;
        B = mkpts0.c;
        N = mkpts0.h;

        if (mconf.dims != 2)
            return -1;

        const bool conf_is_bk = mconf.w == N && mconf.h == B;
        const bool conf_is_kb = mconf.w == B && mconf.h == N;
        if (!conf_is_bk && !conf_is_kb)
            return -1;

        if (has_m_bids)
            return -1;
    }
    else if (mkpts0.dims == 2)
    {
        // Flat matches [N, 2] + conf [N] (+ optional batch ids [N])
        prebatched = false;
        N = mkpts0.h;

        if (mconf.dims != 1 || mconf.w != N)
            return -1;

        if (has_m_bids)
        {
            if (m_bids->dims != 1 || m_bids->w != N)
                return -1;

            B = 0;
            for (int i = 0; i < N; i++)
            {
                int bid = get_m_bid(i);
                if (bid < 0)
                    continue;
                if (bid + 1 > B)
                    B = bid + 1;
            }
            if (B < 1)
                B = 1;
        }
        else
        {
            B = 1;
        }
    }
    else
    {
        return -1;
    }

    Mat out0(2, K, B);
    Mat out1(2, K, B);
    Mat outc(K, B);

    if (out0.empty() || out1.empty() || outc.empty())
        return -100;

    out0.fill(0.f);
    out1.fill(0.f);
    outc.fill(-1.f);

    for (int b = 0; b < B; b++)
    {
        std::vector<std::pair<float, int> > candidates;
        candidates.reserve(N);

        for (int i = 0; i < N; i++)
        {
            int bid = 0;
            if (!prebatched && has_m_bids)
            {
                bid = get_m_bid(i);
            }
            else if (prebatched)
            {
                bid = b;
            }

            if (bid != b)
                continue;

            float conf = 0.f;
            if (prebatched)
            {
                if (mconf.w == N && mconf.h == B)
                {
                    conf = mconf.row(b)[i];
                }
                else
                {
                    conf = mconf.row(i)[b];
                }
            }
            else
            {
                conf = mconf[i];
            }

            if (conf > neg_inf_half)
                candidates.push_back(std::make_pair(conf, i));
        }

        std::sort(candidates.begin(), candidates.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        const int keep = std::min((int)candidates.size(), K);
        float* outc_row = outc.row(b);
        Mat out0_ch = out0.channel(b);
        Mat out1_ch = out1.channel(b);

        for (int k = 0; k < keep; k++)
        {
            const int idx = candidates[k].second;
            const float conf = candidates[k].first;

            const float* p0 = prebatched ? mkpts0.channel(b).row(idx) : mkpts0.row(idx);
            const float* p1 = prebatched ? mkpts1.channel(b).row(idx) : mkpts1.row(idx);

            float* q0 = out0_ch.row(k);
            float* q1 = out1_ch.row(k);
            q0[0] = p0[0];
            q0[1] = p0[1];
            q1[0] = p1[0];
            q1[1] = p1[1];
            outc_row[k] = conf;
        }
    }

    top_blobs.resize(3);
    top_blobs[0] = out0;
    top_blobs[1] = out1;
    top_blobs[2] = outc;

    return 0;
}

} // namespace ncnn

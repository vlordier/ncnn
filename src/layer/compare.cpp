// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "compare.h"

namespace ncnn {

Compare::Compare()
{
    one_blob_only = false;
    support_inplace = false;
}

int Compare::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    if (with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }

    return 0;
}

template<typename Op>
static void compare_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    const Op op;

    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const float* ptr = a;
        const float* ptr1 = b;
        float* outptr = c;

        const int ainc = a.w > 1 ? 1 : 0;
        const int binc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = op(*ptr, *ptr1);
            ptr += ainc;
            ptr1 += binc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const float* ptr = a.row(y < a.h ? y : a.h - 1);
            const float* ptr1 = b.row(y < b.h ? y : b.h - 1);
            float* outptr = c.row(y);

            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = op(*ptr, *ptr1);
                ptr += ainc;
                ptr1 += binc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = c.channel(q);

            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = a.channel(q < a.c ? q : a.c - 1).depth(z < a.d ? z : a.d - 1).row(y < a.h ? y : a.h - 1);
                    const float* ptr1 = b.channel(q < b.c ? q : b.c - 1).depth(z < b.d ? z : b.d - 1).row(y < b.h ? y : b.h - 1);

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(*ptr, *ptr1);
                        ptr += ainc;
                        ptr1 += binc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

template<typename Op>
static void compare_scalar_inplace(Mat& a, float b, const Option& opt)
{
    const Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }
}

struct compare_op_eq
{
    float operator()(const float& x, const float& y) const
    {
        return x == y ? 1.f : 0.f;
    }
};

struct compare_op_ne
{
    float operator()(const float& x, const float& y) const
    {
        return x != y ? 1.f : 0.f;
    }
};

struct compare_op_gt
{
    float operator()(const float& x, const float& y) const
    {
        return x > y ? 1.f : 0.f;
    }
};

struct compare_op_ge
{
    float operator()(const float& x, const float& y) const
    {
        return x >= y ? 1.f : 0.f;
    }
};

struct compare_op_lt
{
    float operator()(const float& x, const float& y) const
    {
        return x < y ? 1.f : 0.f;
    }
};

struct compare_op_le
{
    float operator()(const float& x, const float& y) const
    {
        return x <= y ? 1.f : 0.f;
    }
};

int Compare::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (with_scalar)
    {
        if (bottom_blobs.size() != 1)
            return -1;

        top_blobs[0] = bottom_blobs[0].clone(opt.blob_allocator);
        if (top_blobs[0].empty())
            return -100;

        return forward_inplace(top_blobs[0], opt);
    }

    if (bottom_blobs.size() != 2)
        return -1;

    const Mat& a = bottom_blobs[0];
    const Mat& b = bottom_blobs[1];
    const int outdims = std::max(a.dims, b.dims);

    Mat a2 = a;
    Mat b2 = b;
    if (a.dims < outdims)
    {
        // expand inner axes same as BinaryOp
        if (outdims == 2)
        {
            if (a.w == b.h)
                a2 = a.reshape(1, a.w);
            else
                a2 = a.reshape(a.w, 1);
        }
        if (outdims == 3 && a.dims == 1)
        {
            if (a.w == b.c)
                a2 = a.reshape(1, 1, a.w);
            else
                a2 = a.reshape(a.w, 1, 1);
        }
        if (outdims == 3 && a.dims == 2)
            a2 = a.reshape(1, a.w, a.h);
        if (outdims == 4 && a.dims == 1)
        {
            if (a.w == b.c)
                a2 = a.reshape(1, 1, 1, a.w);
            else
                a2 = a.reshape(a.w, 1, 1, 1);
        }
        if (outdims == 4 && a.dims == 2)
            a2 = a.reshape(1, 1, a.w, a.h);
        if (outdims == 4 && a.dims == 3)
            a2 = a.reshape(1, a.w, a.h, a.c);
    }
    if (b.dims < outdims)
    {
        // expand inner axes same as BinaryOp
        if (outdims == 2)
        {
            if (b.w == a.h)
                b2 = b.reshape(1, b.w);
            else
                b2 = b.reshape(b.w, 1);
        }
        if (outdims == 3 && b.dims == 1)
        {
            if (b.w == a.c)
                b2 = b.reshape(1, 1, b.w);
            else
                b2 = b.reshape(b.w, 1, 1);
        }
        if (outdims == 3 && b.dims == 2)
            b2 = b.reshape(1, b.w, b.h);
        if (outdims == 4 && b.dims == 1)
        {
            if (b.w == a.c)
                b2 = b.reshape(1, 1, 1, b.w);
            else
                b2 = b.reshape(b.w, 1, 1, 1);
        }
        if (outdims == 4 && b.dims == 2)
            b2 = b.reshape(1, 1, b.w, b.h);
        if (outdims == 4 && b.dims == 3)
            b2 = b.reshape(1, b.w, b.h, b.c);
    }

    const int outw = std::max(a2.w, b2.w);
    const int outh = std::max(a2.h, b2.h);
    const int outd = std::max(a2.d, b2.d);
    const int outc = std::max(a2.c, b2.c);

    if (outdims == 1)
        top_blobs[0].create(outw, (size_t)4u, opt.blob_allocator);
    if (outdims == 2)
        top_blobs[0].create(outw, outh, (size_t)4u, opt.blob_allocator);
    if (outdims == 3)
        top_blobs[0].create(outw, outh, outc, (size_t)4u, opt.blob_allocator);
    if (outdims == 4)
        top_blobs[0].create(outw, outh, outd, outc, (size_t)4u, opt.blob_allocator);

    if (top_blobs[0].empty())
        return -100;

    Mat& c = top_blobs[0];

    if (op_type == Operation_EQ) compare_broadcast<compare_op_eq>(a2, b2, c, opt);
    if (op_type == Operation_NE) compare_broadcast<compare_op_ne>(a2, b2, c, opt);
    if (op_type == Operation_GT) compare_broadcast<compare_op_gt>(a2, b2, c, opt);
    if (op_type == Operation_GE) compare_broadcast<compare_op_ge>(a2, b2, c, opt);
    if (op_type == Operation_LT) compare_broadcast<compare_op_lt>(a2, b2, c, opt);
    if (op_type == Operation_LE) compare_broadcast<compare_op_le>(a2, b2, c, opt);

    return 0;
}

int Compare::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (!with_scalar)
        return -1;

    if (op_type == Operation_EQ) compare_scalar_inplace<compare_op_eq>(bottom_top_blob, b, opt);
    if (op_type == Operation_NE) compare_scalar_inplace<compare_op_ne>(bottom_top_blob, b, opt);
    if (op_type == Operation_GT) compare_scalar_inplace<compare_op_gt>(bottom_top_blob, b, opt);
    if (op_type == Operation_GE) compare_scalar_inplace<compare_op_ge>(bottom_top_blob, b, opt);
    if (op_type == Operation_LT) compare_scalar_inplace<compare_op_lt>(bottom_top_blob, b, opt);
    if (op_type == Operation_LE) compare_scalar_inplace<compare_op_le>(bottom_top_blob, b, opt);

    return 0;
}

} // namespace ncnn

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_bias_scale(
    Tensor<Engine0, Layout0> &score,
    Tensor<Engine1, Layout1> &bias,
    const float &scale)
{
    static_assert(Layout0::rank == 2, "Attn_score only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Cluster_bias only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(bias) == size<0>(score));
    CUTE_STATIC_ASSERT_V(size<1>(bias) == size<1>(score));
    #pragma unroll
    for (int mi = 0; mi < size<0>(score); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(score); ++ni)  {
            score(mi, ni) = score(mi, ni) * scale + bias(mi, ni);
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_max(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max)
{
    static_assert(Layout0::rank == 2, "Attn_score only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Row_max only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) - max(mi));
        }
    }
}

template <int kNRows>
struct SoftmaxKVClus {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ SoftmaxKVClus() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, Tensor2 &sB, float softmax_scale) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor bias = make_tensor(sB.data(), flash::convert_layout_acc_rowcol(sB.layout()));

        flash::apply_bias_scale(scores, bias, softmax_scale);
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::apply_max(scores, row_max);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f(scores_max_prev(mi) - scores_max_cur);
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            flash::apply_max(scores, row_max);
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

}  // namespace flash

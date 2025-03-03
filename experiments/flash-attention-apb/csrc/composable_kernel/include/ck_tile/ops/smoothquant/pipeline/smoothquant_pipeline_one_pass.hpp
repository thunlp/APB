// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = SmoothquantPipelineDefaultPolicy>
struct SmoothquantPipelineOnePass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using XScaleDataType  = ck_tile::remove_cvref_t<typename Problem::XScaleDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using QYDataType      = ck_tile::remove_cvref_t<typename Problem::QYDataType>;
    using YScaleDataType  = ck_tile::remove_cvref_t<typename Problem::YScaleDataType>;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockSmoothquantProblem::kPadM
    static constexpr bool kPadN              = Problem::kPadN;
    static constexpr bool UseMax3            = true; // TODO - Move to trait

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_op"; // block per row
        else
            return "wpr_op"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow, typename XScaleWindow, typename QYWindow, typename YScaleWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XScaleWindow& xscale_window_,
                                   YScaleWindow& yscale_window,
                                   QYWindow& qy_window,
                                   ck_tile::index_t,
                                   void* smem) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto xscale_window = make_tile_window(
            xscale_window_, Policy::template MakeXScaleBlockTileDistribution<Problem>());

        auto reduce_absmax_func  = ReduceOp::AbsMax{};
        auto reduce_absmax3_func = [](auto acc_, auto v_0_, auto v_1_) {
            float rtn;
            asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                         : "=v"(rtn)
                         : "v"(acc_), "v"(v_0_), "v"(v_1_));
            return rtn;
        };
        auto reduce_max_func = ReduceOp::Max{};

        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        const auto x      = load_tile(x_window);
        const auto xscale = load_tile(xscale_window);
        auto y            = tile_elementwise_in(
            [&](const auto& a, const auto& b) {
                return type_convert<ComputeDataType>(a) * type_convert<ComputeDataType>(b);
            },
            x,
            xscale);

        // compute absmax, cross-lane->cross-warp
        auto absmax = [&]() {
            constexpr auto x_size_per_row =
                x.get_tile_distribution().get_ys_to_d_descriptor().get_lengths().at(number<1>{});
            if constexpr(UseMax3 && std::is_same_v<ComputeDataType, float> &&
                         x_size_per_row % 2 == 0)
            {
                return block_reduce2d(y,
                                      reduce_absmax_func.GetIdentityValue<ComputeDataType>(),
                                      reduce_absmax3_func,
                                      sequence<1, 2>{});
            }
            else
            {
                return block_reduce2d(
                    y, reduce_absmax_func.GetIdentityValue<ComputeDataType>(), reduce_absmax_func);
            }
        }();
        block_reduce2d_sync(absmax, reduce_max_func);
        block_reduce2d_cross_warp_sync(absmax, smem, reduce_max_func);

        // ex: yscale = absmax / 127 if int8
        auto yscale = tile_elementwise_in(
            [&](const auto& v_) {
                return v_ / type_convert<ComputeDataType>(numeric<QYDataType>::max());
            },
            absmax);
        store_tile(yscale_window, cast_tile<YScaleDataType>(yscale));

        // quantize y to qy
        auto qy = make_static_distributed_tensor<QYDataType>(y.get_tile_distribution());
        sweep_tile(qy, [&](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            auto qy_             = y[idx] / yscale[i_idx];
            qy(idx)              = saturates<QYDataType>{}(qy_);
        });
        store_tile(qy_window, qy);
    }
};
} // namespace ck_tile


// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "add_rmsnorm2d_rdquant_fwd_instance_common.hpp"

// clang-format off
//                                                               rm  rn  tm  tn  vn     pd    x     3p
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, 1,  1,  4,  64, 8,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, 1,  2,  4,  64, 4,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, 1,  4,  4,  64, 2,  true , true, false>>(const S&, A);
template float add_rmsnorm2d_rdquant_fwd_<trait_<ck_tile::fp16_t, 1,  8,  4,  64, 1,  true , true, false>>(const S&, A);
// clang-format on

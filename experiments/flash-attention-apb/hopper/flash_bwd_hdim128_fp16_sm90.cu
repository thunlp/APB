// Copyright (c) 2024, anonymous.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::half_t, 128>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<cutlass::half_t>(params, stream);
}

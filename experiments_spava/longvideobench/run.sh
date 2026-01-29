# We use the global var `DURATION_GROUP` to set the subtask of LongVideoBench
# `DURATION_GROUP` can be set to 15, 60, 600, 3600

# Here are the examples of running the experiments on LongVideoBench
# Qwen-2.5-VL models
# FlashAttn
DURATION_GROUP=15 python testd.py
# XAttn
DURATION_GROUP=15 python testd_x.py
# Sparge
DURATION_GROUP=15 python testd_sparge.py
# SlowFast
DURATION_GROUP=15 python testd_sf.py
# StarAttn
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 teststar.py
# APB
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 testapb.py
# Spava
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 testspava.py

# InternVL-3 models
# FlashAttn
DURATION_GROUP=15 python testd_intern.py
# XAttn
DURATION_GROUP=15 python testd_intern_x.py
# Sparge
DURATION_GROUP=15 python testd_intern_sparge.py
# SlowFast
DURATION_GROUP=15 python testd_intern_sf.py
# StarAttn
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 testd_intern_star.py
# APB
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 testd_intern_apb.py
# Spava
DURATION_GROUP=15 torchrun --nnodes=1 --nproc_per_node=8 testd_intern_apbv.py

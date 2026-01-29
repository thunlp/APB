### Hyperparameters:
# CLARITY: the clarity of input videos, can be set to 360, 720, 1080, and 1440.
# MIN_FNUM: minimum frame number, always set to 16
# MAX_FNUM: maximum frame number, always set to 256
# MODEL: model identifier that appears in the directory of saved results
# NOTE!!! apbv is another name of spava that we use during developing

# Qwen-2.5VL series:
# FullAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" python qwen_full.py
# XAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" python qwen_x.py
# Sparge
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" python qwen_sparge.py
# SlowFast
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" python qwen_sf.py
# ZZRing
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" torchrun --nnodes=1 --nproc_per_node=8 qwen_zzring.py
# StarAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" torchrun --nnodes=1 --nproc_per_node=8 qwen_star.py
# APB
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" torchrun --nnodes=1 --nproc_per_node=8 qwen_apb.py
# Spava
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="qwen-3b" torchrun --nnodes=1 --nproc_per_node=8 qwen_apbv.py

# InternVL-3 series:
# FullAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" python intern_full.py
# XAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" python intern_x.py
# Sparge
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" python intern_sparge.py
# SlowFast
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" python intern_sf.py
# ZZRing
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" torchrun --nnodes=1 --nproc_per_node=8 intern_zzring.py
# StarAttn
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" torchrun --nnodes=1 --nproc_per_node=8 intern_star.py
# APB
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" torchrun --nnodes=1 --nproc_per_node=8 intern_apb.py
# Spava
CLARITY=360 MIN_FNUM=16 MAX_FNUM=256 MODEL="intern-2b" torchrun --nnodes=1 --nproc_per_node=8 intern_apbv.py

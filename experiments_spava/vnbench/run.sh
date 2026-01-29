# For the Qwen-2.5VL models:

cd qwenvl2_5
# FullAttn
bash niah_eval.sh
# XAttn
bash niah_eval_x.sh
# Sparge
bash niah_eval_sparge.sh
# SlowFast
bash niah_eval_sparge.sh
# StarAttn
bash niah_eval_star.sh
# APB
bash niah_eval_apb.sh
# Spava
bash niah_eval_spava.sh

cd ../

# For the InternVL-3 models:
cd internvl3
bash niah_eval.sh
# XAttn
bash niah_eval_x.sh
# Sparge
bash niah_eval_sparge.sh
# SlowFast
bash niah_eval_sparge.sh
# StarAttn
bash niah_eval_star.sh
# APB
bash niah_eval_apb.sh
# Spava
bash niah_eval_spava.sh

cd ../

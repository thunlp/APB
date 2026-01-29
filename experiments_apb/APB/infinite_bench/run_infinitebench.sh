# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

MODEL_PATH="/yourpath"
MAX_SEQ_LEN=131072
NUM_EVAL_EXAMPLES=-1
num_proc=8

TASKS=("longbook_choice_eng" "longbook_qa_eng" "longdialogue_qa_eng" "longbook_qa_chn" "code_debug" "longbook_sum_eng" "number_string" "passkey" "kv_retrieval" "math_find")

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

for task in ${TASKS[@]}; do
echo $task
torchrun --nproc_per_node=${num_proc} "$SCRIPT_DIR/run_infinitebench.py" \
    --task $task \
    --model_name_or_path $MODEL_PATH \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length $MAX_SEQ_LEN \
    --rewrite \
    --trust_remote_code \
    --num_eval_examples $NUM_EVAL_EXAMPLES --topk 1 --starting_layer 0 --attn_type hf \
    --method lring # or star, ulysses
done

# bash run_infinitebench.sh gradientai/Llama-3-8B-Instruct-262k 160000 10 minference

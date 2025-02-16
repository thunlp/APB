# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

MODEL_PATH="/yourpath"
LOCRET_PATH="/yourpath"
MAX_SEQ_LEN=131072
NUM_EVAL_EXAMPLES=-1
num_nodes=2
num_proc=8
master_addr=g72
master_port=12345

TASKS=("longbook_choice_eng" "longbook_qa_eng" "longdialogue_qa_eng" "longbook_qa_chn" "code_debug" "longbook_sum_eng" "number_string" "passkey" "kv_retrieval" "math_find")

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

for task in ${TASKS[@]}; do
echo $task
torchrun --nnodes=${num_nodes} --nproc_per_node=${num_proc} --rdzv_id "apb" --rdzv_backend=c10d --rdzv_endpoint=${master_addr}:${master_port} \
"$SCRIPT_DIR/run_infinitebench_yi_time.py" \
    --task $task \
    --model_name_or_path $MODEL_PATH \
    --locret_path $LOCRET_PATH \
    --data_dir ./data \
    --output_dir ./results_yi \
    --max_seq_length $MAX_SEQ_LEN \
    --rewrite \
    --trust_remote_code \
    --num_eval_examples $NUM_EVAL_EXAMPLES --topk 1 --starting_layer 0 --attn_type hf \
    --method ring# or star, ulysses, lring
done

# bash run_infinitebench.sh gradientai/Llama-3-8B-Instruct-262k 160000 10 minference

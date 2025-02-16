# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

MODEL_PATH="/yourpath"
MAX_SEQ_LEN=131072
NUM_EVAL_EXAMPLES=-1

# Define the tasks to run
# TASKS=("longbook_choice_eng" "longbook_qa_eng" "longdialogue_qa_eng" "longbook_qa_chn" "code_debug" "longbook_sum_eng" "number_string" "passkey")
# TASKS=("kv_retrieval" "math_find")

TASKS=("longbook_choice_eng")


# Set environment variables
export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

# Run each task on a different GPU
for i in ${!TASKS[@]}; do
  task=${TASKS[$i]}
  gpu_id=$((2*i))  # Assign GPU ID based on task index
  
  echo "Running task: $task on GPU $gpu_id"
  
  CUDA_VISIBLE_DEVICES=$gpu_id,$((gpu_id+1)) python "$SCRIPT_DIR/run_infinitebench_minf_time.py" \
      --task $task \
      --model_name_or_path $MODEL_PATH \
      --data_dir ./data \
      --output_dir ./results_yi \
      --max_seq_length $MAX_SEQ_LEN \
      --rewrite \
      --trust_remote_code \
      --num_eval_examples $NUM_EVAL_EXAMPLES --topk 1 --starting_layer 0 --attn_type minference \
      --method minference &
done

# Wait for all background jobs to finish
wait

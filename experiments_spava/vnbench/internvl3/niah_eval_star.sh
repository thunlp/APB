#!/bin/bash
ROOT_DIR="/your_path_prefix/VideoNIAH-main"
CKPT="/your_path_prefix/InternVL3-2B"
ANNOFILE="/your_path_prefix/VideoNIAH-main/VNBench-main-4try.json"
VIDEODIR="/your_path_prefix/VideoNIAH-main/VNBench"
OUTPUT="/your_path_prefix/VideoNIAH-main/results_intern_star"
if [ ! -e $ROOT_DIR ]; then
    echo "The root dir does not exist. Exiting the script."
    exit 1
fi

cd $ROOT_DIR
export PYTHONPATH="./:$PYTHONPATH"
# export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# CUDA_VISIBLE_DEVICES=0 # DEBUG
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"
FRAMES=64
mkdir -p $OUTPUT/video_niah_${FRAMES}

torchrun --nnodes=1 --nproc_per_node=8 internvl3/model_video_niah_star.py \
    --model-path $CKPT \
    --video_dir $VIDEODIR \
    --question_fp $ANNOFILE \
    --output_dir $OUTPUT/video_niah_${FRAMES} \
    --output_name pred \
    --num-chunks 1 \
    --chunk-idx 0 \
    --frames_num $FRAMES 

wait
output_file=$OUTPUT/video_niah_${FRAMES}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
cat $OUTPUT/video_niah_${FRAMES}/${CHUNKS}_0.json >> "$output_file"

python ./internvl3/evaluation_utils.py --result_path $output_file
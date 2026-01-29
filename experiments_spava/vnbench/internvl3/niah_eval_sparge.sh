#!/bin/bash
ROOT_DIR="/your_path_prefix/VideoNIAH-main"
CKPT="/your_path_prefix/InternVL3-2B"
ANNOFILE="/your_path_prefix/VideoNIAH-main/VNBench-main-4try.json"
VIDEODIR="/your_path_prefix/VideoNIAH-main/VNBench"
OUTPUT="/your_path_prefix/VideoNIAH-main/results_intern_sparge"
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

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_ID=${GPULIST[$IDX]}  # Note: Zsh arrays are 1-indexed by default
    # if [[ "$IDX" -ne 6 ]]; then
    #     continue
    # fi
    echo "Running on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python internvl3/model_video_niah_sparge.py \
    --model-path $CKPT \
    --video_dir $VIDEODIR \
    --question_fp $ANNOFILE \
    --output_dir $OUTPUT/video_niah_${FRAMES} \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $(($IDX - 1)) \
    --frames_num $FRAMES &

done
wait
output_file=$OUTPUT/video_niah_${FRAMES}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq -1 $((CHUNKS-2))); do
    cat $OUTPUT/video_niah_${FRAMES}/${CHUNKS}_${IDX}.json >> "$output_file"
done

python ./internvl3/evaluation_utils.py --result_path $output_file
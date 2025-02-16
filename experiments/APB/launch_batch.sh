
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_PATH="/yourpath"


BENCHMARK="ruler"

# Benchmark config setup
# ======================

if [[ $BENCHMARK == "ruler" ]]; then

    PROMPT_CONFIG="llama3" # Select from: ruler/data/template.py
    SCRIPT="run_ruler.py"

elif [[ $BENCHMARK == "babilong" ]]; then

    PROMPT_CONFIG="llama3" # Select from: babilong/template.py
    SCRIPT="run_babilong.py"

else
    echo "Invalid Benchmark: ${BENCHMARK}"
    exit 1
fi

# Launch Evaluation
# =================

## The `-np` config shared below is tested for the Llama-3.1-8B-Instruct model on 8 A100 GPUs

## methods: dense, ring, star, lring


# Global Attention
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a dense -pc $PROMPT_CONFIG -l 16384
python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a lring -pc $PROMPT_CONFIG -l 131072 -np 8 --time --batch
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a lring -pc $PROMPT_CONFIG -l 131072 -np 4 --time
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 16384 -l 131072 -np 8
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 16384 -l 131072 -np 8 --time --breakdown
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ring -pc $PROMPT_CONFIG -l 131072 -np 8 --time --breakdown
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ulysses -pc $PROMPT_CONFIG -l 131072 -np 8 --time --breakdown
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a flash -pc $PROMPT_CONFIG -l 131072 --time --breakdown
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a minf -pc $PROMPT_CONFIG -l 131072 --time --batch
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ring -pc $PROMPT_CONFIG -l 16384 -np 8
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a dense -pc $PROMPT_CONFIG -l 16384 32768 65536
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a ring -pc $PROMPT_CONFIG -l 131072 -np 8

# Star Attention
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 4096 -l 16384 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 8192 -l 32768 -np 4
# python $SCRIPT -p $MODEL_PATH -n $MODEL_NAME -a star -pc $PROMPT_CONFIG -bs 16384 -l 65536 -np 4

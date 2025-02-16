
# Reproduction Instructions of APB


Here, we offer the intructions on reproducing the evaluations of APB with baselines selected in our paper on InfiniteBench and RULER.

## Setup

First, create a conda environment by `environment.yml`

```
conda env create -f environment.yml
```

Then, install `flash-attention-apb` and `ring-flash-attention`. These two packages are modified from [flash-attention](https://github.com/Dao-AILab/flash-attention) and [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)

```
cd flash-attention-apb
pip install -e .
cd ../ring-flash-attention-main
pip install -e .
cd ..
```

## Data Preparation

### InifiniteBench

For the next step, download the data of infinitebench following [https://github.com/OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench%E2%80%B8) and save it to `APB/infinite_bench/_data/`.

### RULER

```
cd APB/ruler
bash download_data.sh
cd ../../
```

## Training of the Retaining Heads

Finally, train the retaining heads of Locret by following [https://arxiv.org/abs/2410.01805](https://arxiv.org/abs/2410.01805). Then, use `APB/convert.py` to filter out the parameters of the retaining heads. Save it as your locret path.

## Conducting the Benchmarks

### RULER

Run launch.sh in *APB*. Note that `lring` is an alias of our method APB.

```
cd APB
bash launch.sh
```

You can execute the commands in launch.sh to evaluate a specific baseline under different LLM.

### For infinitebench

All the experiment startup commands can be found in the bash files under `APB/infinite_bench/`.
For example, to evaluate the speed of StarAttn under `Qwen-2.5-14B-instruct`, you can run `run_infinitebench_qwen_time.sh` and set `--method star` in the command.

```
cd APB/infinite_bench
bash run_infinitebench_qwen_time.sh
```

### Benchmarking across Input Length.

First, you need to change the block size and passing length in `ring-flash-attention-main/ring-flash-attn/lring_flash_attn.py` and `APB/lring/modeling_llama_lring.py` for different input length. Then run `launch_len.sh` in *APB*.

```
cd APB
bash launch_len.sh
```


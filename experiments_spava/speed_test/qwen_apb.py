import os
import re
import torch
import random
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from qwen2_5_vl.modeling_qwen2_5_vl_apb import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from longvideobench import LongVideoBenchDataset
import time
import numpy as np

# 环境设置
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def collate_fn(x):
    return x[0]

class VideoInferencePipeline:
    def __init__(self, model_path, dataset_path, json_file, rank, world_size, clarity):
        # 初始化模型
        self.rank = rank
        self.world_size = world_size
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{rank}",  # 每卡独立加载
            attn_implementation="flash_attention_2"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.frame_num = -1
        assert clarity in [360, 720, 1080, 1440]
        if clarity == 360:
            self.width, self.height = 640, 360
        elif clarity == 720:
            self.width, self.height = 1280, 720
        elif clarity == 1080:
            self.width, self.height = 1920, 1080
        elif clarity == 1440:
            self.width, self.height = 2560, 1440

    def process_single_sample(self, data):
        """处理单个样本的完整流程（修改为支持DP）"""
        # 预处理
        messages, choices, questions = preprocess_sample(data["inputs"])
        answer = data["correct_choice"]

        # 生成输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        flat_list = [item for sublist in video_inputs for item in sublist]
        if len(flat_list) / 2 < dist.get_world_size():
            raise ValueError("video too short")
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(f"cuda:{self.rank}")

        query_str = "".join(questions) + "<|im_end|>\n<|im_start|>assistant\n"
        queries = self.processor.tokenizer(query_str, add_special_tokens=False, return_tensors='pt')['input_ids'].to(inputs.input_ids.device)
        position_ids = torch.arange(0, inputs.input_ids.shape[-1]).unsqueeze(0).to(inputs.input_ids.device)
        try:
            generated_ids = self.model.apbv_generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                position_ids=position_ids,
                context_len=position_ids.shape[-1] - queries.shape[-1],
                pixel_values_videos=inputs.pixel_values_videos,
                video_grid_thw=inputs.video_grid_thw,
                second_per_grid_ts=inputs.second_per_grid_ts,
                stop_ids=[self.processor.tokenizer.eos_token_id],
                sys_prompt_len=14,
            )
            output_text = self.processor.decode(generated_ids)
        except:
            print("error")
            output_text = ""

        return output_text, choices, answer

    def run_evaluation(self, frame_num=-1):
        self.frame_num = frame_num
        messages, query_str = preprocess_sample(self.width, self.height, self.frame_num)

        # 生成输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(f"cuda:{self.rank}")

        query_str = query_str + "<|im_end|>\n<|im_start|>assistant\n"
        queries = self.processor.tokenizer(query_str, add_special_tokens=False, return_tensors='pt')['input_ids'].to(inputs.input_ids.device)
        position_ids = torch.arange(0, inputs.input_ids.shape[-1]).unsqueeze(0).to(inputs.input_ids.device)

        with torch.inference_mode():
            for _ in range(5):
                generated_ids = self.model.apbv_generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    position_ids=position_ids,
                    context_len=position_ids.shape[-1] - queries.shape[-1],
                    pixel_values_videos=inputs.pixel_values_videos,
                    video_grid_thw=inputs.video_grid_thw,
                    second_per_grid_ts=inputs.second_per_grid_ts,
                    stop_ids=[self.processor.tokenizer.eos_token_id],
                    sys_prompt_len=14,
                    max_new_tokens=1,
                )
            
            dist.barrier()
            beg = time.time()
            for _ in range(5):
                generated_ids = self.model.apbv_generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    position_ids=position_ids,
                    context_len=position_ids.shape[-1] - queries.shape[-1],
                    pixel_values_videos=inputs.pixel_values_videos,
                    video_grid_thw=inputs.video_grid_thw,
                    second_per_grid_ts=inputs.second_per_grid_ts,
                    stop_ids=[self.processor.tokenizer.eos_token_id],
                    sys_prompt_len=14,
                    max_new_tokens=1,
                )
            dist.barrier()
            end = time.time()

        return 5 / (end - beg)


def preprocess_sample(width, height, frame_num):
    video = [
        Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8), mode='RGB') for _ in range(frame_num)
    ]
    content = [
        {"type": "video", "video": video},
        {"type": "text", "text": "What is in this video?"}
    ]
    
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages, "What is in this video?"

def parse_multi_choice_response(response, all_choices):
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]
import json
def main(rank, world_size):
    # 初始化分布式环境
    
    clarity = int(os.environ['CLARITY'])
    min_frame_num = int(os.environ['MIN_FNUM'])
    max_frame_num = int(os.environ['MAX_FNUM'])
    model_name = os.environ['MODEL']

    # 初始化流水线
    pipeline = VideoInferencePipeline(
        model_path="/your_path_prefix/Qwen2.5-VL-3B-Instruct",
        dataset_path="/your_path_prefix/apb-v/LongVideoBench",
        json_file="lvb_val.json",
        rank=rank,
        world_size=world_size,
        clarity=clarity,
    )
    
    if dist.get_rank() == 0:
        os.makedirs(model_name, exist_ok=True)
        fp = open(f"{model_name}/apb_{clarity}p_{min_frame_num}_{max_frame_num}.jsonl", "w")

    # 运行评估
    for fn in range(min_frame_num, max_frame_num + 1, 8):
        speed = pipeline.run_evaluation(frame_num=fn)
        data = {
            "model": model_name,
            "clarity": clarity,
            "frame_num": fn,
            "method": "apb",
            "speed": speed,
        }
        data_str = json.dumps(data)
        if dist.get_rank() == 0:
            fp.write(f"{data_str}\n")
            fp.flush()
    
    if dist.get_rank() == 0:
        fp.close()
    dist.barrier()
    


def init_distributed():
    """Initialize the distributed environment."""

    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'[APB-V.init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    return rank, world_size


if __name__ == "__main__":
    rank, world_size = init_distributed()
    main(rank, world_size)
    # world_size = 8
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
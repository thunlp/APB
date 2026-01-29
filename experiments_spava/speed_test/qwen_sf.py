import os
import re
import torch
import random
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor
from qwen2_5_vl.modeling_qwen2_5_vl_sf import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from longvideobench import LongVideoBenchDataset
import numpy as np
import time

# 环境设置
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def collate_fn(x):
    return x[0]

class VideoInferencePipeline:
    def __init__(self, model_path, dataset_path, json_file, rank, world_size, clarity):
        # 初始化模型
        self.rank = rank
        self.world_size = world_size
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            # "/your_path_prefix/apb-v/checkpoints_ltr_1e-3/checkpoint_5000",
            torch_dtype=torch.bfloat16,
            device_map=f"auto",  # 每卡独立加载
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
        messages, choices = preprocess_sample(data["inputs"])
        answer = data["correct_choice"]

        # 生成输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        flat_list = [item for sublist in video_inputs for item in sublist]
        if len(flat_list) / 2 < 8:
            raise ValueError("video too short")
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(f"cuda:{self.rank}")
        # 模型推理
        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
            # 后处理
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        except:
            print("error")
            output_text = [""]
        
        return output_text[0], choices, answer

    def run_evaluation(self, frame_num=-1):
        self.frame_num = frame_num
        # 预处理
        messages = preprocess_sample(self.width, self.height, self.frame_num)

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

        with torch.inference_mode():
            for _ in range(5):
                self.model(**inputs, use_cache=True)
            

            beg = time.time()
            for _ in range(5):
                self.model(**inputs, use_cache=True)
            end = time.time()
        
        print(f"{frame_num}, {((end - beg) / 5):.2f}")
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
    
    return messages

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
    # setup(rank, world_size)
    
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
    
    os.makedirs(model_name, exist_ok=True)
    fp = open(f"{model_name}/sf_{clarity}p_{min_frame_num}_{max_frame_num}.jsonl", "w")

    # 运行评估
    for fn in range(min_frame_num, max_frame_num + 1, 8):
        speed = pipeline.run_evaluation(frame_num=fn)
        data = {
            "model": model_name,
            "clarity": clarity,
            "frame_num": fn,
            "method": "sf",
            "speed": speed,
        }
        data_str = json.dumps(data)
        fp.write(f"{data_str}\n")
        fp.flush()
    

    fp.close()


if __name__ == "__main__":
    # world_size = 8
    world_size = 1
    main(0, 1)
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
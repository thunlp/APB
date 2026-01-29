import os
import re
import torch
import random
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from internvl3.modeling_internvl_chat import InternVLChatModel
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
        self.model = InternVLChatModel.from_pretrained(
            model_path,
            # "/your_path_prefix/apb-v/checkpoints_ltr_1e-3/checkpoint_5000",
            torch_dtype=torch.bfloat16,
            device_map=f"auto",  # 每卡独立加载
            attn_implementation="flash_attention_2"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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



    def run_evaluation(self, frame_num=-1):
        self.frame_num = frame_num
        # 预处理
        pixel_values, num_patches_list = fake_video_loader(num_segments=self.frame_num)

        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + "What is in this video?"

        generation_config = dict(max_new_tokens=1, do_sample=False)

        for _ in range(5):
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=False)

        beg = time.time()
        for _ in range(5):
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=False)
        end = time.time()
    
        print(f"{frame_num}, {((end - beg) / 5):.2f}")
        return 5 / (end - beg)


import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Function to extract frames from video
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=64):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def fake_video_loader(video_path=None, bound=None, input_size=448, max_num=1, num_segments=64):
    # 模拟 FPS 和帧数
    fps = 30.0
    max_frame = 1000  # 假设视频有 300 帧

    # 模拟获取帧索引
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    pixel_values_list = []
    num_patches_list = []

    transform = build_transform(input_size=input_size)

    for _ in frame_indices:
        # 创建一个随机RGB图像（模拟帧）
        img_array = np.uint8(np.random.rand(input_size, input_size, 3) * 255)
        img = Image.fromarray(img_array).convert('RGB')

        # 伪造切图输出
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)

        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)

        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

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
        model_path="/your_path_prefix/InternVL3-2B",
        dataset_path="/your_path_prefix/apb-v/LongVideoBench",
        json_file="lvb_val.json",
        rank=rank,
        world_size=world_size,
        clarity=clarity,
    )
    
    os.makedirs(model_name, exist_ok=True)
    fp = open(f"{model_name}/full_{clarity}p_{min_frame_num}_{max_frame_num}.jsonl", "w")

    # 运行评估
    for fn in range(min_frame_num, max_frame_num + 1, 8):
        speed = pipeline.run_evaluation(frame_num=fn)
        data = {
            "model": model_name,
            "clarity": clarity,
            "frame_num": fn,
            "method": "full",
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
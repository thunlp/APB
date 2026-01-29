import argparse
import torch

from operator import attrgetter
from internvl3.modeling_internvl_chat_sf import InternVLChatModel
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
from transformers import AutoConfig
import json
import os

import math
from tqdm import tqdm
from decord import VideoReader, cpu

import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument('--question_fp', help='Path to the question file.', required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--frames_num", type=int, default=4)
    return parser.parse_args()

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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=64):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import copy
from decord import VideoReader, cpu

class VideoQADataset(Dataset):
    def __init__(self, question_dict, video_dir, frames_num, device):
        self.data = question_dict
        self.video_dir = video_dir
        self.frames_num = frames_num
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q_dict = self.data[idx]
        q_uid = q_dict['video'].split('/')[-1].replace('.mp4', '')
        if not os.path.exists(q_dict["video"]):
            video_path = os.path.join(self.video_dir, q_dict["video"])
        else:
            video_path = q_dict["video"]

        question0 = q_dict['question']
        options = q_dict['options']
        question = f"{question0}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly."
        ret_dict = copy.deepcopy(q_dict)
        ret_dict['bare_question'] = question
        # Process prompt.
        qs = question

        if os.path.exists(video_path):
            pixel_values, num_patches_list = load_video(video_path, num_segments=self.frames_num, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + qs


        ret_dict['pixel_values'] = pixel_values
        ret_dict['num_patches_list'] = num_patches_list
        ret_dict['qs'] = question

        return ret_dict

from torch.utils.data import DataLoader

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"

    model = InternVLChatModel.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        device_map=f"auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained)


    model.eval()

    if '.jsonl' in args.question_fp:
        question_dict = [json.loads(q) for q in open(os.path.expanduser(args.question_fp), "r")]
    else:
        question_dict = json.load(open(args.question_fp))
    question_dict = get_chunk(question_dict, args.num_chunks, args.chunk_idx)
    # question_dict = [q for q in question_dict if q['type']=='ord_edit1']


    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    qa_dataset = VideoQADataset(question_dict, args.video_dir, args.frames_num, model.device)

    qa_dataloader = DataLoader(qa_dataset, batch_size=None, shuffle=False, num_workers=8, prefetch_factor=2)

    index = 0
    for q_dict in tqdm(qa_dataloader):
        pixel_values = q_dict['pixel_values']
        qs = q_dict['qs']
        bare_question = q_dict['bare_question']
        num_patches_list = q_dict['num_patches_list']
        options = q_dict['options']

        generation_config = dict(max_new_tokens=5, do_sample=False)

        response = model.chat(tokenizer, pixel_values, qs, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=False)

        outputs = response.strip()

        outputs = outputs.strip()
        gt = chr(options.index(q_dict['gt']) + ord('A'))
        inf_res = {"video_path": q_dict['video'],
                    "prompt": qs,
                    "pred": outputs,
                    "gt": gt, 
                    "task_type": q_dict['type'],
                    "try": q_dict['try'],
                    "model_id": model_name}

        ans_file.write(json.dumps(inf_res) + "\n")

        ans_file.flush()

    ans_file.close()

import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    run_inference(args)
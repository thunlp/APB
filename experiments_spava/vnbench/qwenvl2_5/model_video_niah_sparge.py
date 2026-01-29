import argparse
import torch

from operator import attrgetter
from qwen2_5_vl.modeling_qwen2_5_vl_sparge import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

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


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

def load_video_1fps(video_path, max_frames_num, load_fps=1):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))

    fps = vr.get_avg_fps()
    total_frame_num = len(vr)
    duration_seconds = int(total_frame_num / fps)

    if duration_seconds * load_fps > max_frames_num:
        max_frames_num = int(duration_seconds * load_fps)

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        device_map=f"auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(pretrained)


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

    index = 0
    for q_dict in tqdm(question_dict):
        q_uid = q_dict['video'].split('/')[-1].replace('.mp4', '')
        if not os.path.exists(q_dict["video"]):
            video_path = os.path.join(args.video_dir, q_dict["video"])
        else:
            video_path = q_dict["video"]

        # Check if the video exists
        if os.path.exists(video_path):
            video_frames = load_video(video_path, args.frames_num)
            video_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in video_frames]

        question0 = q_dict['question']
        options = q_dict['options']
        question = f"{question0}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly."
        # Process prompt.
        qs = question


        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        flat_list = [item for sublist in video_inputs for item in sublist]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(model.device)


        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

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


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from qwen_vl_utils import process_vision_info

import json

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16, duration_group=3600):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    if duration_group in [15, 60]:
        num_frames = min(max_num_frames, int(duration * fps))
    else:
        num_frames = min(max_num_frames, int(duration))
    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    #print("frame:", frame_timestamp)
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            #print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            #print("leaving out subtitle:", start, end)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        #print(frame_timestamp)
        interleaved_list.append(frame)
        
    return interleaved_list
    

def preprocess_sample(sample):
    content = []
    current_video = []
    choices = []
    
    questions = []
    end = False

    for item in sample:
        if isinstance(item, str):
            if current_video:
                content.append({"type": "video", "video": current_video})
                current_video = []
            content.append({"type": "text", "text": item})
            if item.startswith("Question") or end:
                questions.append(item)
                end = True
            if len(item) >= 2 and item[:2] in ['A.', 'B.', 'C.', 'D.', 'E.']:
                choices.append(item)
        elif isinstance(item, Image.Image):
            current_video.append(item)
        else:
            raise ValueError(f"不支持的数据类型: {type(item)}")
    
    if current_video:
        content.append({"type": "video", "video": current_video})
    
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages, choices, questions

class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                 duration_group=None,
                 processor=None,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        self.duration_group = duration_group
        if duration_group is not None:
            self.data = [d for d in self.data if d['duration_group'] == duration_group]
        self.max_num_frames = max_num_frames
        self.processor = processor
        
        
        
    def __getitem__(self, index):
        di = self.data[index]
        
        if self.max_num_frames == 0:
            ### No subtitles, no frames        
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
        if self.max_num_frames == -1:
            ### All subtitles, no frames
            with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
                subtitles = json.load(f)
            inputs = insert_subtitles(subtitles)
            inputs += ["Question: " + di["question"]]
            inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
            inputs += ["Answer with the option's letter from the given choices directly."]
            return {"inputs": inputs, "correct_choice": chr(ord("A")+di["correct_choice"]), "id": di["id"]}
            
        frames, frame_timestamps = load_video(os.path.join(self.data_path, "videos", di["video_path"]), di["duration"], max_num_frames=self.max_num_frames, duration_group=self.duration_group)
        
            
        with open(os.path.join(self.data_path, "subtitles", di["subtitle_path"])) as f:
            subtitles = json.load(f)
        inputs = []
        if self.insert_text:
            inputs = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, di["starting_timestamp_for_subtitles"], di["duration"])
        else:
            inputs = frames

        ##### YOU MAY MODIFY THE FOLLOWING PART TO ADAPT TO YOUR MODEL #####
        inputs += ["Question: " + di["question"]]
        inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
        inputs += ["Answer with the option's letter from the given choices directly."]
        ##### YOU MAY MODIFY THE PREVIOUS PART TO ADAPT TO YOUR MODEL #####

        





        ##### CORRECT CHOICE WILL BE "@" FOR TEST SET SAMPLES #####
        return {"inputs": inputs, "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"]}

        # data = {"inputs": inputs, "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"]}

        # messages, choices, questions = preprocess_sample(data["inputs"])
        # answer = data["correct_choice"]

        # # 生成输入
        # text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs, video_kwargs = process_vision_info(
        #     messages, return_video_kwargs=True
        # )

        # flat_list = [item for sublist in video_inputs for item in sublist]
        # if len(flat_list) / 2 < dist.get_world_size():
        #     raise ValueError("video too short")
        
        # inputs = self.processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        #     **video_kwargs
        # ).to(f"cuda:{dist.get_rank()}")

        # query_str = "".join(questions) + "<|im_end|>\n<|im_start|>assistant\n"
        # queries = self.processor.tokenizer(query_str, add_special_tokens=False, return_tensors='pt')['input_ids'].to(inputs.input_ids.device)
        # position_ids = torch.arange(0, inputs.input_ids.shape[-1]).unsqueeze(0).to(inputs.input_ids.device)
        # return inputs, position_ids

    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]
        
if __name__ == "__main__":
    db = LongVideoBenchDataset("../", "lvb_val.json")
    for i in range(10):
        print([ele for ele in db[i]["inputs"] if not isinstance(ele, str)])
                     

            
            
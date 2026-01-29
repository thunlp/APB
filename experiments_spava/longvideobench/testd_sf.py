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
# from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from longvideobench import LongVideoBenchDataset
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
    def __init__(self, model_path, dataset_path, json_file, rank, world_size):
        # 初始化模型
        self.rank = rank
        self.world_size = world_size
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            # "/your_path_prefix/apb-v/checkpoints_ltr_1e-3/checkpoint_5000",
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.rank}",  # 每卡独立加载
            attn_implementation="flash_attention_2"
        )
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        
        self.processor = AutoProcessor.from_pretrained(model_path)

        duration_group = int(os.getenv("DURATION_GROUP", 15))
        # 创建数据集和分布式采样器
        self.dataset = LongVideoBenchDataset(dataset_path, json_file, max_num_frames=64, duration_group=duration_group, processor=self.processor, insert_text=False)

        # self.dataset.data = self.dataset.data[74:] # TODO!!!

        self.sampler = DistributedSampler(
            self.dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=4,
            prefetch_factor=4,
            sampler=self.sampler,
            collate_fn=collate_fn,
        )

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


        # device = torch.device(f"cuda:{self.rank}")
        # inputs, position_ids, queries, choices, answer = data
        # # inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # inputs = inputs.to(device)
        # position_ids = position_ids.to(device)
        # queries = queries.to(device)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to(f"cuda:{self.rank}")
        # breakpoint()
        # 模型推理
        # try:
        beg_t = time.time()
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        end_t = time.time()
        
        # 后处理
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        # except:
        #     print("error")
            # output_text = [""]
        print(output_text[0])
        return output_text[0], choices, answer, end_t - beg_t

    def run_evaluation(self):
        """分布式评估流程"""
        total_correct = torch.tensor(0, dtype=torch.int64, device=f"cuda:{self.rank}")
        local_correct = 0
        
        # 仅主卡显示进度条
        # if self.rank == 0:
        pbar = tqdm(total=len(self.dataset)//self.world_size, desc=f"Rank {self.rank} Processing")
        output_texts = []
        too_short = 0
        total_time = 0
        total_cnt = 0
        for data in self.dataloader:
            # try:
            output_text, choices, answer, time_t = self.process_single_sample(data)
            total_time += time_t
            total_cnt += 1
            output_texts.append(output_text)
            pred = parse_multi_choice_response(output_text, choices)
            local_correct += int(pred == answer)
            
            # if self.rank == 0:
            pbar.update(1)
            pbar.set_postfix({"current_acc": local_correct/pbar.n})
                    
            # except Exception as e:
            #     too_short += 1
            #     print(f"Rank {self.rank} error: {str(e)}")
            #     continue
        
        # if self.rank == 0:
        #     for o in output_texts:
        #         print("="*50)
        #         print(o)
        #     print("="*50)
        
        # 汇总所有GPU结果
        total_correct += torch.tensor(local_correct, dtype=torch.int64, device=f"cuda:{self.rank}")
        print(total_correct)

        print(f"avg. time: {(total_time / total_cnt):.2f} s/req")

        # torch.cuda.synchronize()
        # dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        # dist.barrier()        

        if self.rank == 0:
            pbar.close()
            print(total_correct.item(), len(self.dataset), too_short)
            return total_correct.item() / len(self.dataset)
        else:
            return None

def preprocess_sample(sample):
    content = []
    current_video = []
    choices = []
    
    for item in sample:
        if isinstance(item, str):
            if current_video:
                content.append({"type": "video", "video": current_video})
                current_video = []
            content.append({"type": "text", "text": item})
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
    
    return messages, choices

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

def main(rank, world_size):
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 初始化流水线
    pipeline = VideoInferencePipeline(
        model_path="/your_path_prefix/Qwen2.5-VL-7B-Instruct",
        dataset_path="/your_path_prefix/LongVideoBench",
        json_file="lvb_val.json",
        rank=rank,
        world_size=world_size
    )
    
    # 运行评估
    accuracy = pipeline.run_evaluation()
    
    # 仅主卡输出结果
    if rank == 0:
        print(f"\nFinal Accuracy: {accuracy:.2%}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 8
    # world_size = 1
    # main(0, 1)
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
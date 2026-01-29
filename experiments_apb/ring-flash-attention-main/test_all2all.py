import torch
import torch.distributed as dist
import os
from ring_flash_attn.utils import All2AllComm

# 初始化分布式环境
def init_distributed_environment():
    dist.init_process_group(backend='nccl')

# 初始化分布式环境
init_distributed_environment()

# 获取当前进程的rank和world_size
rank = dist.get_rank()
world_size = dist.get_world_size()

# 创建一个All2AllComm实例
process_group = dist.new_group(list(range(world_size)))
comm = All2AllComm(process_group)

# 假设我们有一个形状为 [bsz, n_head, seq_len, 2*head_dim] 的张量
bsz = 8
n_head = 12
seq_len = 100
head_dim = 64

# 创建输入张量
input_tensor = torch.randn(bsz, n_head, seq_len, 2 * head_dim, dtype=torch.float32).to(f"cuda:{rank}")
input_tensor[:] = rank  # 每个进程的输入是它自己的 rank

# 执行all2all通信
comm.all2all(input_tensor)

# 等待通信完成
comm.wait()

# 获取通信结果
output_tensors = comm.data()

# 打印结果
print(f"Rank {rank} received data")

# 检查结果是否正确
for i in range(world_size):
    expected_shape = (bsz, n_head, seq_len, 2 * head_dim)
    output_tensor = output_tensors[i]
    if output_tensor.shape != expected_shape:
        raise ValueError(f"Output tensor shape {output_tensor.shape} does not match expected shape {expected_shape}")

# 验证每个进程接收到的数据是否正确
for i in range(world_size):
    std_tensor = torch.zeros_like(input_tensor)
    std_tensor[:] = i
    if not torch.allclose(output_tensors[i], std_tensor):
        print(output_tensors[i])
        raise ValueError(f"Rank {rank} received incorrect data from rank {i}")

if rank == 0:
    print(len(output_tensors))
    print(output_tensors)

print(f"Rank {rank} passed the test.")

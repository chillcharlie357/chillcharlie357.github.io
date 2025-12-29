---
title: PyTorch 分布式机制详解
date: 2025-12-29 15:51
modified: 2025-12-29 16:09
tags:
  - pytorch
  - distributed
  - NCCL
  - AIInfra
  - 多GPU训练
categories:
  - 技术分享
excerpt: 深入理解 torch.distributed 在 Nano-vLLM 中的应用
mathjax: true
comment: true
---

# PyTorch 分布式机制详解

> 深入理解 torch.distributed 在 Nano-vLLM 中的应用

---

## 目录

1. [概述](#概述)
2. [Barrier 详解](#barrier-详解)
3. [通信后端](#通信后端)
4. [集体通信原语](#集体通信原语)
5. [在 Nano-vLLM 中的应用](#在-nano-vllm-中的应用)
6. [实战示例](#实战示例)
7. [最佳实践](#最佳实践)

---

## 概述

### 什么是 PyTorch 分布式？

**PyTorch 分布式 (`torch.distributed`)** 是 PyTorch 提供的多进程/多机训练和推理框架，支持：

- ✅ **数据并行**：每个 GPU 处理不同数据
- ✅ **张量并行**：模型切分到多个 GPU（Nano-vLLM 使用）
- ✅ **流水线并行**：模型层分布到不同 GPU
- ✅ **多机训练**：跨多台机器的分布式执行

### 核心概念

```
进程组 (Process Group)
   └─ 一组协同工作的进程

世界大小 (world_size)
   └─ 总进程数（例如：4 个 GPU = world_size=4）

秩 (rank)
   └─ 进程的唯一标识（0, 1, 2, ...）

本地秩 (local_rank)
   └─ 单机内的进程编号

后端 (backend)
   └─ 通信实现：NCCL, Gloo, MPI
```

### 初始化示例

```python
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(
    backend="nccl",                        # 使用 NCCL 后端
    init_method="tcp://localhost:2333",    # 主进程地址
    world_size=4,                          # 4 个进程
    rank=0                                 # 当前进程的 rank
)

# 设置当前使用的 GPU
torch.cuda.set_device(rank)
```

---

## Barrier 详解

### 什么是 Barrier？

**Barrier（屏障）** 是一种**同步原语**，确保所有进程都到达同一点后，才继续执行。

### 直观理解

```
就像赛跑的起跑线：

运动员 1: 到达起跑线 → 等待...
运动员 2: 到达起跑线 → 等待...
运动员 3: 到达起跑线 → 等待...
运动员 4: 到达起跑线 → 等待...
            ↓
     所有人都到了！
            ↓
        发令枪响
            ↓
     所有人同时起跑
```

### 技术原理

```python
# 进程 0
dist.barrier()
  ├─ 进入阻塞状态
  ├─ 等待其他进程调用 barrier()
  └─ 收到所有进程到达信号 → 解除阻塞

# 进程 1
dist.barrier()
  ├─ 进入阻塞状态
  ├─ 等待其他进程调用 barrier()
  └─ 收到所有进程到达信号 → 解除阻塞

# ... 所有进程同理
```

### 使用场景

| 场景 | 作用 | 示例 |
|------|------|------|
| **资源初始化** | 确保共享资源创建完成 | 共享内存、文件创建 |
| **数据同步** | 确保所有进程数据就绪 | 模型加载、数据预处理 |
| **性能测量** | 同步开始/结束时间 | 准确测量推理延迟 |
| **资源清理** | 确保所有进程释放资源 | 删除共享内存、关闭文件 |

### 代码示例

```python
import torch.distributed as dist

def train():
    # 每个进程加载数据
    data = load_data(rank)

    # 等待所有进程加载完成
    dist.barrier()

    # 开始训练（所有进程同时开始）
    for epoch in range(epochs):
        train_step(data)
```

---

## 通信后端

PyTorch 支持多种通信后端，各有适用场景。

### NCCL (NVIDIA Collective Communications Library)

**最适合 GPU 多卡/多机场景**

```python
dist.init_process_group("nccl", ...)
```

**特点**：
- ✅ **性能最优**：专为 GPU 优化，支持 NVLink
- ✅ **自动优化**：根据拓扑结构选择最优路径
- ✅ **支持 GPU Direct RDMA**：绕过 CPU，直接 GPU-GPU 传输
- ❌ **仅限 NVIDIA GPU**

**Nano-vLLM 使用**：

```python
# nanovllm/engine/model_runner.py:26
dist.init_process_group("nccl", "tcp://localhost:2333",
                        world_size=self.world_size, rank=rank)
```

### Gloo

**适用于 CPU 或少量 GPU**

```python
dist.init_process_group("gloo", ...)
```

**特点**：
- ✅ **跨平台**：支持 Linux/Windows/macOS
- ✅ **CPU/GPU 混合**：可在 CPU 上运行
- ❌ **性能较低**：相比 NCCL 慢 2-5x
- ❌ **不支持 NVLink**

**适用场景**：
- CPU 集群训练
- 开发/调试（单机多卡）
- 跨操作系统部署

### MPI

**适用于超算/科研场景**

```python
dist.init_process_group("mpi", ...)
```

**特点**：
- ✅ **生态成熟**：MPI 标准实现
- ✅ **超算优化**：支持 InfiniBand 等高速网络
- ✅ **灵活配置**：支持丰富的进程拓扑
- ❌ **配置复杂**：需要安装 MPI 实现

**适用场景**：
- 国家超算中心
- 科研计算集群
- 自定义网络拓扑

### 后端对比

| 后端 | 硬件 | 带宽 | 延迟 | 配置难度 | 推荐场景 |
|------|------|------|------|---------|---------|
| **NCCL** | NVIDIA GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | 生产环境 GPU 集群 |
| **Gloo** | CPU/GPU | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 开发/调试/CPU 训练 |
| **MPI** | 任意 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 超算/科研 |

---

## 集体通信原语

PyTorch 提供多种集体通信（Collective Communication）原语。

### 1. Barrier - 同步点

```python
dist.barrier()
```

**作用**：同步所有进程

**示例**：
```python
# Rank 0
create_shared_memory()
dist.barrier()  # 等待其他 ranks
print("Shared memory ready")

# Rank 1
dist.barrier()  # 等待 rank 0 创建
open_shared_memory()  # 安全访问
```

### 2. Broadcast - 广播

```python
dist.broadcast(tensor, src=0)
```

**作用**：将 `src` 进程的数据广播到所有进程

```
广播前:
Rank 0: [1, 2, 3]
Rank 1: [0, 0, 0]
Rank 2: [0, 0, 0]

dist.broadcast(tensor, src=0)

广播后:
Rank 0: [1, 2, 3]
Rank 1: [1, 2, 3]
Rank 2: [1, 2, 3]
```

**示例**：模型参数初始化
```python
if rank == 0:
    model = initialize_model()

# 广播模型参数到所有 GPUs
for param in model.parameters():
    dist.broadcast(param.data, src=0)
```

### 3. All-Reduce - 全局归约

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

**作用**：对所有进程的数据执行归约操作，结果返回给所有进程

```
归约前:
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

归约后:
Rank 0: [12, 15, 18]  ← 所有 ranks 都得到结果
Rank 1: [12, 15, 18]
Rank 2: [12, 15, 18]
```

**归约操作**：
- `ReduceOp.SUM`：求和（最常用）
- `ReduceOp.PRODUCT`：乘积
- `ReduceOp.MAX`：最大值
- `ReduceOp.MIN`：最小值
- `ReduceOp.AVG`：平均值

**示例**：张量并行梯度同步
```python
# nanovllm/layers/linear.py:152
class RowParallelLinear:
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(y)  # ← 求和所有 GPU 的部分结果
        return y
```

### 4. Reduce - 归约

```python
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
```

**作用**：对所有进程的数据执行归约，结果只发送到 `dst` 进程

```
归约前:
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

归约后:
Rank 0: [12, 15, 18]  ← 只有 rank 0 得到结果
Rank 1: [4, 5, 6]     ← 其他 ranks 保持不变
Rank 2: [7, 8, 9]
```

**示例**：汇总损失
```python
# 计算平均损失
dist.reduce(loss, dst=0, op=dist.RedceOp.SUM)
if rank == 0:
    loss /= world_size
    print(f"Average loss: {loss}")
```

### 5. All-Gather - 全收集

```python
dist.all_gather(tensor_list, tensor)
```

**作用**：收集所有进程的数据到每个进程

```
收集前:
Rank 0: [1, 1, 1]
Rank 1: [2, 2, 2]
Rank 2: [3, 3, 3]

dist.all_gather(tensor_list, tensor)

收集后:
Rank 0: [[1,1,1], [2,2,2], [3,3,3]]  ← 所有 ranks 都得到完整数据
Rank 1: [[1,1,1], [2,2,2], [3,3,3]]
Rank 2: [[1,1,1], [2,2,2], [3,3,3]]
```

**示例**：收集各 GPU 的预测结果
```python
# 每个进程预测一部分数据
local_predictions = model(local_data)

# 收集所有进程的预测
all_predictions = [torch.empty_like(local_predictions) for _ in range(world_size)]
dist.all_gather(all_predictions, local_predictions)
```

### 6. Scatter - 分发

```python
dist.scatter(tensor, scatter_list, src=0)
```

**作用**：将数据分发到不同进程

```
分发前:
Rank 0: [[1,1,1], [2,2,2], [3,3,3]]
Rank 1: []
Rank 2: []

dist.scatter(tensor, scatter_list, src=0)

分发后:
Rank 0: [1, 1, 1]  ← 每个进程得到一部分
Rank 1: [2, 2, 2]
Rank 2: [3, 3, 3]
```

### 性能对比

| 操作 | 数据量 | 时间复杂度 |
|------|--------|-----------|
| **Barrier** | 0 字节 | O(log P) |
| **Broadcast** | N 字节 | O(N) |
| **All-Reduce** | N 字节 | O(N) |
| **Reduce** | N 字节 | O(N) |
| **All-Gather** | N×P 字节 | O(N×P) |
| **Scatter** | N×P 字节 | O(N×P) |

注：P = 进程数

**说明**：
- 实际性能取决于硬件配置（NVLink/PCIe/网络）
- NCCL 在 NVLink 上可达到最高性能
- 延迟和带宽因 GPU 型号（A100/H100）和配置而异

---

## 在 Nano-vLLM 中的应用

### 1. 初始化阶段

**文件**：`nanovllm/engine/model_runner.py:16-48`

```python
def __init__(self, config: Config, rank: int, event: Event):
    # 初始化 NCCL 进程组
    dist.init_process_group("nccl", "tcp://localhost:2333",
                            world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)

    # ... 模型加载 ...

    # 多 GPU 同步初始化
    if self.world_size > 1:
        if rank == 0:
            # 主进程创建共享内存
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()  # ← 等待 worker GPUs 准备好
        else:
            # Worker 进程等待共享内存创建完成
            dist.barrier()  # ← 确保共享内存已创建
            self.shm = SharedMemory(name="nanovllm")
            self.loop()  # 进入事件循环
```

**为什么需要两个 barrier？**

```
Rank 0 创建共享内存需要时间
   ↓
dist.barrier() # Rank 0 等待
   ↓
所有 Ranks 都到达屏障
   ↓
Rank 1, 2, ... 安全打开共享内存
```

### 2. 退出清理阶段

**文件**：`nanovllm/engine/model_runner.py:50-59`

```python
def exit(self):
    if self.world_size > 1:
        self.shm.close()      # 所有 ranks 关闭共享内存
        dist.barrier()        # ← 确保所有 ranks 都关闭
        if self.rank == 0:
            self.shm.unlink() # 只有 rank 0 删除共享内存
    # ... 其他清理 ...
    dist.destroy_process_group() # 销毁进程组
```

**为什么需要 barrier？**

```
没有 barrier：
Rank 0: 关闭 → 删除共享内存 ❌
Rank 1: 仍在使用 → 访问已删除的内存 ❌ 崩溃！

有 barrier：
Rank 0: 关闭 → 等待...
Rank 1: 关闭 → 等待...
         ↓ 所有 ranks 都关闭
Rank 0: 删除共享内存 ✅ (安全)
```

### 3. 张量并行通信

**文件**：`nanovllm/layers/linear.py`

```python
class RowParallelLinear(nn.Module):
    """行并行：输出按维度切分，需要 All-Reduce"""
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(y)  # ← 隐式 NCCL 通信
        return y

class ColumnParallelLinear(nn.Module):
    """列并行：权重按维度切分，无需通信"""
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        return y  # 不需要通信，结果已经是切分的
```

**张量并行通信流程**：

```
输入 X: [batch, hidden]

GPU 0:         GPU 1:
  Y₀ = X @ W₀     Y₁ = X @ W₁
  [部分结果]      [部分结果]
       │              │
       └── all_reduce ─┘
              │
         Y = Y₀ + Y₁
        [完整结果]
```

---

## 实战示例

### 示例 1：基本的分布式初始化

```python
import torch
import torch.distributed as dist
import os

def setup():
    """初始化分布式环境"""
    # 从环境变量读取配置（标准做法）
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # GPU 用 NCCL
        init_method="env://"  # 从环境变量读取
    )

    # 设置当前 GPU
    torch.cuda.set_device(local_rank)

    return rank, world_size

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

# 使用示例
if __name__ == "__main__":
    rank, world_size = setup()
    print(f"Rank {rank} initialized")

    # 你的训练/推理代码
    # ...

    cleanup()
```

**启动命令**：
```bash
# 单机 4 卡
torchrun --nproc_per_node=4 script.py

# 多机 2x4 卡
torchrun --nnodes=2 --nproc_per_node=4 \
         --master_addr=10.0.0.1 --master_port=29500 \
         script.py
```

### 示例 2：使用 Barrier 同步

```python
import torch.distributed as dist
import torch

def synchronized_training():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1. 每个进程加载数据
    if rank == 0:
        print("Rank 0: Loading data...")
    data = load_data_for_rank(rank)

    # 2. 等待所有进程加载完成
    dist.barrier()
    if rank == 0:
        print("All ranks loaded data")

    # 3. 同时开始训练
    for epoch in range(10):
        train_one_epoch(data)

        # 4. 同步检查点
        if epoch % 5 == 0:
            dist.barrier()
            if rank == 0:
                print(f"Epoch {epoch} completed on all ranks")
```

### 示例 3：All-Reduce 梯度同步

```python
import torch.distributed as dist
import torch.nn as nn

def train_step(model, data, target):
    """分布式训练步骤"""
    # 1. 前向传播
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    # 2. 反向传播（本地梯度）
    loss.backward()

    # 3. 同步梯度（数据并行）
    for param in model.parameters():
        if param.grad is not None:
            # All-Reduce 梯度：求和所有进程的梯度
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # 除以进程数得到平均梯度
            param.grad.div_(dist.get_world_size())

    # 4. 更新参数
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

### 示例 4：分布式推理

```python
import torch.distributed as dist
import torch

def distributed_inference(model, dataloader):
    """分布式推理：每个 GPU 处理不同数据"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model.eval()
    results = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 只处理分配给当前 rank 的数据
            if i % world_size != rank:
                continue

            # 推理
            output = model(batch)

            # 收集结果（可选）
            results.append(output.cpu())

    # All-Gather：收集所有 rank 的结果
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    # 只有 rank 0 保存完整结果
    if rank == 0:
        # 合并所有结果
        final_results = merge_results(all_results)
        return final_results
    return None
```

### 示例 5：监控性能

```python
import torch.distributed as dist
import time

def benchmark_distributed(func, num_iterations=100):
    """测量分布式操作的性能"""
    rank = dist.get_rank()

    # 预热
    for _ in range(10):
        func()

    # 同步开始
    dist.barrier()
    start_time = time.time()

    # 执行测试
    for _ in range(num_iterations):
        func()

    # 同步结束
    dist.barrier()
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) / num_iterations

    # 只有 rank 0 打印结果
    if rank == 0:
        print(f"Average time: {avg_time * 1000:.2f} ms")

    return avg_time
```

---

## 最佳实践

### 1. 使用环境变量配置

**推荐**：
```python
dist.init_process_group(
    backend="nccl",
    init_method="env://"  # 从环境变量读取
)
```

**不推荐**：
```python
dist.init_process_group(
    backend="nccl",
    init_method="tcp://192.168.1.1:2333",  # 硬编码 IP
    world_size=4,
    rank=0
)
```

**环境变量**：
```bash
RANK=0
WORLD_SIZE=4
LOCAL_RANK=0
MASTER_ADDR=10.0.0.1
MASTER_PORT=29500
```

### 2. 始终清理进程组

```python
try:
    # 你的代码
    train()
finally:
    # 确保清理
    dist.destroy_process_group()
```

### 3. 使用 Barrier 确保同步

```python
# 模型保存
if rank == 0:
    save_checkpoint(model)

# 等待保存完成
dist.barrier()

# 所有 rank 可以安全继续
```

### 4. 避免死锁

**错误示例**：
```python
# 只有 rank 0 执行
if rank == 0:
    dist.all_reduce(tensor)  # ❌ 其他 ranks 永远等待！
```

**正确示例**：
```python
# 所有 ranks 都执行
dist.all_reduce(tensor)  # ✅
```

### 5. 选择合适的后端

| 场景 | 推荐后端 |
|------|---------|
| GPU 多卡训练 | `nccl` |
| GPU 多机训练 | `nccl` |
| CPU 训练 | `gloo` |
| 开发调试 | `gloo` |
| 超算环境 | `mpi` |

### 6. 性能优化

```python
# 使用 pin_memory 加速 CPU→GPU 传输
data = torch.tensor(data, pin_memory=True)
data = data.cuda(non_blocking=True)

# 合并小操作
# ❌ 多次通信
for x in tensors:
    dist.all_reduce(x)

# ✅ 一次通信
big_tensor = torch.cat(tensors)
dist.all_reduce(big_tensor)
```

---

## 常见问题

### Q1: 什么时候需要使用 Barrier？

**需要**：
- ✅ 创建/删除共享资源（共享内存、文件）
- ✅ 同步性能测量
- ✅ 确保数据准备就绪

**不需要**：
- ❌ All-Reduce/All-Gather 等集体通信（已内置同步）
- ❌ 独立的数据处理

### Q2: NCCL vs Gloo 如何选择？

| 条件 | 选择 |
|------|------|
| NVIDIA GPU | NCCL |
| CPU 训练 | Gloo |
| 开发/调试 | Gloo |
| 生产环境 | NCCL |

### Q3: 如何调试分布式代码？

```python
import os

# 只让 rank 0 打印
if dist.get_rank() == 0:
    print("Debug info")

# 每个 rank 打印到不同文件
with open(f"rank_{dist.get_rank()}.log", "w") as f:
    f.write("Debug info")
```

---

## 参考资料

- **PyTorch 官方文档**：https://pytorch.org/docs/stable/distributed.html
- **NCCL 文档**：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- **PyTorch 分布式教程**：https://pytorch.org/tutorials/beginner/dist_overview.html
- **分布式训练最佳实践**：https://pytorch.org/tutorials/recipes/recipes/distributed_training.html

---

## 总结

### 关键要点

1. **Barrier 是同步原语**，确保所有进程到达同一点
2. **NCCL 是 GPU 首选**，性能最优
3. **集体通信自动同步**，无需额外 barrier
4. **环境变量配置**，避免硬编码
5. **始终清理资源**，防止泄漏

### 在 Nano-vLLM 中的作用

```python
# 初始化：确保共享内存创建
dist.barrier()

# 张量并行：隐式 All-Reduce
dist.all_reduce(y)

# 退出：确保安全清理
dist.barrier()
```

**核心价值**：
- ✅ 多 GPU 协同工作
- ✅ 张量并行支持大模型
- ✅ 自动通信优化
- ✅ 简洁的 API 设计

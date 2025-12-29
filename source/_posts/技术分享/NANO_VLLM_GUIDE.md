---
aliases:
tags:
  - 技术分享
  - vllm
  - AIInfra
categories:
sticky:
thumbnail:
cover:
excerpt: false
mathjax: true
comment: true
title: Nano-vLLM 源码解析
date: 2025-12-25 16:12
modified: 2025-12-29 11:12
---



# Nano-vLLM 源码解析

> 从原理到实践：深入理解轻量级 LLM 推理引擎

本文档整合了 Nano-vLLM 的核心概念详解和与 vLLM 的对比分析，帮助你全面理解这个轻量级推理引擎。

---

# 目录

1. [概述与对比](#概述与对比)
2. [核心概念详解](#核心概念详解)
3. [使用场景](#使用场景)
4. [学习路径](#学习路径)

---

# 项目结构

## 目录树

```
nanovllm/
├── engine/              # 推理引擎
│   ├── llm_engine.py    # 主引擎（协调调度+执行）
│   ├── scheduler.py     # 调度器（Prefill/Decode 两阶段调度）
│   ├── block_manager.py # Block 管理器（PagedAttention + Prefix Caching）
│   ├── model_runner.py  # 模型执行器（CUDA Graph + 张量并行）
│   └── sequence.py      # 序列状态（WAITING/RUNNING/FINISHED）
│
├── layers/              # 基础层抽象
│   ├── linear.py        # 线性层（张量并行）
│   ├── attention.py     # Flash Attention 封装
│   ├── embed_head.py    # Embedding & LM Head
│   ├── layernorm.py     # RMSNorm
│   ├── activation.py    # SiLU
│   ├── rotary_embedding.py  # RoPE 位置编码
│   └── sampler.py       # 采样策略
│
├── models/              # 模型架构
│   └── qwen3.py         # Qwen3（当前唯一支持的模型）
│
├── utils/               # 工具模块
│   ├── context.py       # 上下文管理（Thread-local 存储）
│   └── loader.py        # 权重加载器
│
├── config.py            # 配置类
├── llm.py               # 对外接口（继承 LLMEngine）
└── sampling_params.py   # 采样参数
```

## 架构层次

```
┌─────────────────────────────────────────┐
│           LLM (对外接口)                  │
│         nanovllm/llm.py                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        LLMEngine (核心引擎)              │
│     engine/llm_engine.py                 │
│  ┌──────────────┬──────────────────┐    │
│  │              │                   │    │
│  ▼              ▼                   ▼    │
│ Scheduler   BlockManager    ModelRunner │
│  调度器       Block管理        模型执行   │
└─────────────────────────────────────┘
                  │
        ┌─────────▼──────────┐
        │   基础层 (layers)  │
        │  Linear/Attention  │
        │   RMSNorm/RoPE     │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  模型 (models)     │
        │    Qwen3           │
        └────────────────────┘
```

## 核心模块说明

### 推理引擎 (`engine/`)

| 模块 | 职责 |
|------|------|
| **Scheduler** | 两阶段调度：Prefill 阶段处理新请求，Decode 阶段继续生成，内存不足时抢占 |
| **BlockManager** | PagedAttention 实现：分配/释放 Block，Prefix Caching 通过哈希去重复用计算 |
| **ModelRunner** | 模型执行：Prefill 用 varlen attention，Decode 用 kvcache attention，CUDA Graph 优化 |
| **Sequence** | 序列状态管理：跟踪 token 数、block_table、缓存状态 |

### 基础层 (`layers/`)

| 模块 | 职责 |
|------|------|
| **Linear** | 张量并行线性层（ColumnParallel, RowParallel, QKVParallel） |
| **Attention** | Flash Attention 封装：Prefill 用 `flash_attn_varlen_func`，Decode 用 `flash_attn_with_kvcache` |
| **RotaryEmbedding** | RoPE 位置编码计算 |
| **Sampler** | 采样策略：Greedy, Top-K, Top-P, Nucleus |

### 模型 (`models/`)

| 模型 | 架构特点 |
|------|---------|
| **Qwen3** | QKV 融合投影 + Gated MLP (SiLU) + Pre-norm (RMSNorm) |

---

# 概述与对比

## 什么是 Nano-vLLM？

Nano-vLLM 是一个**轻量级的 vLLM 实现**，用约 1,200 行 Python 代码实现了生产级推理引擎的核心功能。

### 快速对比

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| **定位** | 生产级 LLM 推理引擎 | 教学级/轻量级实现 |
| **代码量** | ~100,000+ 行 | ~1,200 行 |
| **开发目标** | 性能、功能、稳定性 | 可读性、学习性 |
| **性能** | 基准 | 相当或略优 (1434 vs 1361 tok/s) |

### 类比理解

```
vLLM      = Linux 内核（生产级，功能完整，复杂）
Nano-vLLM = xv6（教学版操作系统，简洁清晰）
```

---

## 共同点（核心设计）

### 1. PagedAttention 内存管理

**两者都使用相同的核心理念**：将 KV 缓存切分为固定大小的 Block，类似操作系统的虚拟内存。

```python
# 核心数据结构
class Block:
    block_id: int      # 物理块 ID
    ref_count: int     # 引用计数
    hash: int          # 用于 prefix caching

block_table: List[int]  # 逻辑块 → 物理块映射
```

**优势**：
- ✅ 减少内存碎片
- ✅ 提高内存利用率
- ✅ 支持动态批处理

### 2. Prefix Caching（前缀缓存）

**相同前缀的请求共享 KV 缓存**。

```python
# 都使用哈希去重
hash = xxhash.xxh64(token_ids)
if hash in cache:
    reuse_cached_block()
```

**效果**：
- 相同前缀的请求共享 KV cache
- 减少重复计算
- 提升 Throughput（可达 100x 加速）

### 3. 两阶段调度

```python
# 两者都采用
while waiting:
    schedule_prefill()   # 批量处理新请求
while running:
    schedule_decode()    # 继续生成
```

### 4. Flash Attention 集成

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# Prefill
flash_attn_varlen_func(q, k, v, ...)

# Decode
flash_attn_with_kvcache(q, k_cache, v_cache, ...)
```

### 5. 张量并行（Tensor Parallelism）

都支持多 GPU 并行，Q/K/V 投影按头分割，输出层合并结果。

---

## 主要差异

### 1. 代码规模和复杂度

| 维度 | vLLM | Nano-vLLM |
|------|------|-----------|
| **总代码量** | ~100,000+ 行 | ~1,200 行 |
| **核心引擎** | ~10,000+ 行 | ~500 行 |
| **模型支持** | 20+ 架构 | 1 架构 (Qwen3) |
| **抽象层次** | 多层抽象 | 扁平化设计 |

**vLLM 复杂度来源**：
```python
# vLLM 有复杂的抽象层
LLM → LLMEngine → Scheduler → ModelExecutor → ModelRunner
         ↓            ↓              ↓
   WorkerPool   BlockSpaceManager  CacheEngine
```

**Nano-vLLM 简化设计**：
```python
# Nano-vLLM 直接扁平化
LLM → LLMEngine
         ↓
    Scheduler + ModelRunner
```

### 2. 模型支持

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| **支持模型** | LLaMA, Mistral, Qwen, GPT-2, Falcon 等 20+ | 仅 Qwen3 |
| **扩展方式** | 注册式架构，易扩展 | 需手动添加模型文件 |
| **权重加载** | 自动映射 + 通用加载器 | 手动定义 `packed_modules_mapping` |

### 3. 调度策略

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| **调度算法** | Maximize Throughput / Priority | 简单 FIFO |
| **抢占策略** | 复杂（考虑 token 预算、优先级） | 简单抢占最后 |
| **批次构建** | 多种策略（连续、优先级等） | 单一策略 |

### 4. 内存管理

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| **块大小** | 可配置（默认 16） | 固定 256 |
| **缓存策略** | 多种（CyclicHashCache 等） | 单一哈希缓存 |
| **缓存淘汰** | LRU/LFU 等策略 | 简单引用计数归零 |

### 5. 功能对比

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| **流式输出** | ✅ | ❌ |
| **OpenAI API** | ✅ | ❌ |
| **异步接口** | ✅ | ❌ |
| **Docker 支持** | ✅ | ❌ |
| **Kubernetes** | ✅ | ❌ |
| **监控（Prometheus）** | ✅ | ❌ |
| **多机分布式** | ✅ | ❌ |

---

# 核心概念详解

本节详细解答 Nano-vLLM 的 6 个核心问题。

---

## 1. PagedAttention 如何解决内存碎片？

### 问题背景

传统 LLM 推理中，每个序列需要连续的 KV 缓存：

```
传统方式:
序列A (500 tokens)  [████████████████████████████]  固定分配
序列B (100 tokens)  [████████]                    浪费 400 tokens
序列C (300 tokens)  [████████████████████]        浪费 200 tokens
```

**问题**：
- ❌ 序列长度不确定，需要预分配最大长度
- ❌ 短序列浪费内存，长序列可能不够用
- ❌ 内存碎片严重，无法有效利用

### 解决方案：分页内存管理

类似操作系统的虚拟内存，将 KV 缓存切分为固定大小的 **Block**（默认 256 tokens）：

```
PagedAttention:
物理内存池:
[Block 0][Block 1][Block 2][Block 3][Block 4][Block 5]

序列A → [Block 0][Block 1][Block 4]  (分散存储)
序列B → [Block 2]
序列C → [Block 3][Block 5]
```

### 核心代码

**Block 定义** (`nanovllm/engine/block_manager.py:8-24`):

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id      # 物理块 ID
        self.ref_count = 0            # 引用计数
        self.hash = -1                # 用于 prefix caching
        self.token_ids = []           # 存储的 token（调试用）
```

**Block 管理器** (`nanovllm/engine/block_manager.py:26-113`):

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size                    # 256 tokens
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块
        self.used_block_ids: set[int] = set()           # 已使用块
```

**序列的块表** (`nanovllm/engine/sequence.py:14-84`):

```python
class Sequence:
    block_table = []           # 逻辑块 → 物理块的映射
    # 例如: [0, 1, 4] 表示
    #   逻辑块 0 → 物理块 0
    #   逻辑块 1 → 物理块 1
    #   逻辑块 2 → 物理块 4
```

### 分配流程

```python
# 伪代码：为序列分配块
def allocate(seq):
    for i in range(seq.num_blocks):              # 需要多少个块
        block_id = self.free_block_ids.popleft() # 取一个空闲块
        seq.block_table.append(block_id)         # 记录映射
        self.used_block_ids.add(block_id)        # 标记使用
```

### 优势

| 传统方式 | PagedAttention |
|---------|----------------|
| 连续内存，浪费严重 | 按需分配，利用率高 |
| 无法共享内存 | 块可共享（prefix caching） |
| 内存碎片化 | 统一管理，无碎片 |

---

## 2. Prefix Caching 如何提升性能？

### 核心思想

**相同前缀的请求共享 KV 缓存**。

例如：
```
请求1: "解释什么是深度学习？"
请求2: "解释什么是深度学习？它的应用有哪些？"
       ↑^^^^^^^^^^^^^^^^^^^ 共享这部分
```

### 实现机制

**1. 块哈希计算** (`nanovllm/engine/block_manager.py:36-41`):

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 链式哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**2. 分配时检查缓存** (`nanovllm/engine/block_manager.py:59-82`):

```python
def allocate(self, seq: Sequence):
    h = -1  # 前一个块的哈希
    cache_miss = False

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)  # 获取第 i 个块的 tokens

        # 计算当前块的哈希（链式）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

        # 检查是否已存在相同哈希的块
        block_id = self.hash_to_block_id.get(h, -1)

        # 缓存未命中或内容不匹配
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 复用缓存块！
            seq.num_cached_tokens += self.block_size  # 标记已缓存
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1  # 增加引用计数
            else:
                block = self._allocate_block(block_id)

        # 更新哈希表
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

        seq.block_table.append(block_id)
```

### 流程图

```
新请求: [1, 2, 3, 4, 5, 6, 7, 8]

Block 分配:
┌─────────┬──────────┬────────────┬─────────────┐
│ Block 0 │ Block 1  │ Block 2    │ Block 3     │
│ [1,2,3] │ [4,5,6]  │ [7]        │ (未满)      │
│ Hash: A │ Hash: B  │ Hash: C    │             │
└─────────┴──────────┴────────────┴─────────────┘
           ↓
检查哈希表: {A: 0, B: 1}  # Block 0 和 Block 1 已存在

结果:
┌─────────┬──────────┬────────────┐
│ 复用    │ 复用     │ 分配新的      │
│ Block 0 │ Block 5  │ Block 3    │
│ (+ref)  │ (+ref)   │  [7,8]     │
└─────────┴──────────┴────────────┘

跳过计算: Block 0 和 Block 1 的 KV cache 直接复用！
```

### 性能提升

```
场景：100 个请求都有相同的前缀 512 tokens

无 Prefix Caching:
- 每个请求都计算 512 tokens 的 attention
- 总计算量: 100 × 512 = 51,200 tokens

有 Prefix Caching:
- 第一个请求计算 512 tokens
- 后续 99 个请求直接复用
- 总计算量: 512 + 99 × 0 = 512 tokens

加速比: 100x！
```

---

## 3. Prefill 和 Decode 的区别是什么？

### 两种阶段

LLM 生成分为两个阶段：

```
输入: "解释量子计算"

Prefill (处理输入):
一次处理所有输入 tokens
[解释, 量子, 计算] → 并行计算 → 生成第一个 token

Decode (生成输出):
逐个生成 tokens
[是] → [一种] → [利用] → [量子] → ... (串行)
```

### 核心区别

| 特性 | Prefill | Decode |
|------|---------|--------|
| **输入长度** | 长（可能 >1000 tokens） | 短（每次 1 token） |
| **并行度** | 所有 tokens 并行处理 | 串行生成 |
| **计算模式** | 矩阵乘法大 batch | 小 batch，高频率 |
| **Attention** | Flash Attn Varlen | Flash Attn with KV Cache |
| **优化手段** | - | CUDA Graph |

### 代码实现

**调度逻辑** (`nanovllm/engine/scheduler.py:24-58`):

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []

    # === 阶段 1: Prefill ===
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查资源
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
            or not self.block_manager.can_allocate(seq)):
            break

        # 分配内存，添加到批次
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True  # ← True 表示 Prefill

    # === 阶段 2: Decode ===
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查是否可以追加
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 抢占其他序列
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    return scheduled_seqs, False  # ← False 表示 Decode
```

### 数据形状对比

```python
# Prefill
input_ids.shape = [total_tokens]  # 例如: [2560] (10个序列，平均256 tokens)
q.shape = [total_tokens, num_heads, head_dim]
cu_seqlens_q = [0, 200, 450, 680, ...]  # 累积边界

# Decode
input_ids.shape = [batch_size]  # 例如: [32] (32个序列，每个1 token)
q.shape = [batch_size, 1, num_heads, head_dim]
context_lens = [120, 256, 512, ...]  # 每个序列的长度
```

#### 详细解释

**为什么 Prefill 使用扁平化张量？**

```
3 个序列的 Prefill:
序列1: [A, B, C]     (3 tokens)
序列2: [D, E]        (2 tokens)
序列3: [F, G, H, I]  (4 tokens)

扁平化拼接:
input_ids = [A, B, C, D, E, F, G, H, I]  # shape: [9]
             序列1     序列2   序列3

累积边界 (cu_seqlens_q):
cu_seqlens_q = [0, 3, 5, 9]
               ↑  ↑  ↑  ↑
               |  |  |  └─ 总长度
               |  |  └──── 序列2结束
               |  └─────── 序列1结束
               └────────── 起始点

通过 cu_seqlens_q 提取序列:
序列1: input_ids[0:3]   → [A, B, C]
序列2: input_ids[3:5]   → [D, E]
序列3: input_ids[5:9]   → [F, G, H, I]
```

**优势：**
- ✅ 大矩阵乘法，GPU 利用率高
- ✅ Flash Attention varlen 支持变长序列
- ✅ 减少 Python 循环开销

**为什么 Decode 使用批处理张量？**

```
3 个序列的 Decode (每个序列只生成 1 个新 token):
序列1: 已有 120 tokens，生成第 121 个
序列2: 已有 256 tokens，生成第 257 个
序列3: 已有 512 tokens，生成第 513 个

批处理:
input_ids = [token_121, token_257, token_513]  # shape: [3]
q.shape = [3, 1, num_heads, head_dim]
            ↑  ↑
            │  └─ 序列维度（只有 1 个新 token）
            └─── 批次维度（3 个序列）

context_lens = [120, 256, 512]  # 告诉 attention 每个序列有多长
```

**Flash Attention 如何使用 context_lens？**

```python
# 伪代码
for i in range(batch_size):
    seq_len = context_lens[i]  # 例如: 120

    # 从 KV cache 读取前 120 个位置
    k_cache[:seq_len]  # shape: [120, num_heads, head_dim]
    v_cache[:seq_len]

    # 计算新 token 与历史 120 个 token 的 attention
    attn(q[i], k_cache[:seq_len], v_cache[:seq_len])
```

**关键区别总结：**

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入方式** | 所有 tokens 扁平化拼接 | 每个序列 1 个 token |
| **形状** | `[total_tokens]` | `[batch_size]` |
| **Q 形状** | `[total_tokens, num_heads, head_dim]` | `[batch_size, 1, num_heads, head_dim]` |
| **边界信息** | `cu_seqlens_q` (累积边界) | `context_lens` (序列长度) |
| **内存访问** | 连续读取全部输入 | 从 KV cache 读取历史 |
| **计算特点** | 一次性处理长序列 | 逐步生成，频繁调用 |

### 资源需求差异

```
Prefill  → 计算密集型 (Compute-bound)
Decode   → 内存带宽密集型 (Memory-bandwidth-bound)
```

**Prefill 为什么计算密集？**

- 处理大量 tokens（可能 1000+），一次矩阵乘法计算 Attention
- 计算复杂度：O(n²)，n 是序列长度
- 数据读一次，大量计算
- **瓶颈**：GPU 计算单元

**Decode 为什么内存带宽密集？**

- 每次只处理 1 个新 token，但要读取整个 KV cache（可能 512+ tokens）
- 计算复杂度：O(n)，n 是序列长度
- 大量数据读取（KV cache），少量计算
- **瓶颈**：内存带宽

**硬件选择建议**：

| 场景 | 阶段 | 推荐硬件 | 关键指标 |
|------|------|---------|---------|
| 长文本生成 | Decode 为主 | H200/B200 | 高内存带宽 (H200: 4.8 TB/s, B200: 8 TB/s) |
| 长文本理解 | Prefill 为主 | H100 | 高算力 (FP8: 3,958 TFLOPS) |
| 平衡型 | 两者兼有 | A100 | 算力 + 带宽平衡 (FP16: 312 TFLOPS, 带宽: 2 TB/s) |

**数据来源**：
- NVIDIA H100/H200/A100 官方规格表[[1]](https://www.nvidia.com/en-us/data-center/h100/)[[2]](https://www.nvidia.com/en-us/data-center/a100/)
- H200 vs H100 vs A100 性能对比[[3]](https://xconnectglobal.com/2025/11/06/nvidia-a100-vs-h100-vs-h200/)
- B200 GPU 规格数据库[[4]](https://www.techpowerup.com/gpu-specs/b200.c4210)

---

## 4. CUDA Graph 如何减少开销？

### 问题：CPU 开销

每次模型推理都需要：
1. CPU 分配内存
2. CPU 设置 CUDA kernel
3. GPU 执行计算
4. CPU 清理

对于 Decode 阶段（每次 1 token，高频执行），CPU 开销占比很高。

### 解决方案：CUDA Graph

**预先捕获整个计算图**，后续只需重放，无需重复设置。

```
传统方式 (每次推理):
CPU: 分配内存 → 设置 kernel → 启动 → 清理
GPU:                    ← 执行 →

CUDA Graph:
CPU: 捕获一次计算图
后续: 重放 (无额外开销)
GPU: ← 执行 →
```

### 实现步骤

**1. 预分配缓冲区** (`nanovllm/engine/model_runner.py:217-228`):

```python
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 预分配最大可能的内存
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
```

**2. 分层捕获** (`nanovllm/engine/model_runner.py:229-242`):

```python
    # 定义要捕获的 batch size
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # 从大到小捕获（共享内存池）
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # Warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 捕获
        with torch.cuda.graph(graph, pool=self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        if self.graph_pool is None:
            self.graph_pool = graph.pool()  # 共享内存池

        self.graphs[bs] = graph
```

**3. 运行时重放** (`nanovllm/engine/model_runner.py:189-206`):

```python
def run_model(self, input_ids, positions, is_prefill):
    # 判断是否使用 CUDA Graph
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill 或超限，使用 eager 模式
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Decode，使用 CUDA Graph
        bs = input_ids.size(0)
        # 找到合适的 batch size
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

        # 更新输入数据
        self.graph_vars["input_ids"][:bs] = input_ids
        self.graph_vars["positions"][:bs] = positions
        self.graph_vars["slot_mapping"][:bs] = slot_mapping
        # ...

        # 重放计算图
        graph.replay()

        return self.model.compute_logits(self.graph_vars["outputs"][:bs])
```

### 性能提升

**NVIDIA 官方基准数据** (2024年9月测量):

对于 100 个节点的 CUDA Graph（典型 LLM decode 场景）:

| 指标 | CUDA 11.8 | CUDA 12.6 | 改善 |
|------|-----------|-----------|------|
| **重复启动 CPU 开销** | 25 μs | 15 μs | **40% ↓** |
| **端到端时间** | 69 μs | 55 μs | **20% ↓** |

**关键发现**:
- 重复启动开销从线性增长（2μs + 200ns/节点）降至几乎恒定（**2.5μs + ~1ns/节点**）
- 对于 10 节点图：CPU 开销从 ~4μs 降至 **~2.5μs**
- 对于 1025 节点图：CPU 开销从 278μs 降至 **~175μs** (37% 改善)

**LLM Decode 场景示意** (单次 token 生成):

```
无 CUDA Graph (多次 kernel launch):
- CPU 累计开销: ~25-50μs (5-10 个 kernels × 5μs)
- GPU 计算: ~100μs
- 总时间: ~125-150μs
- CPU 占比: 20-33%

有 CUDA Graph (一次 graph replay):
- CPU 开销: ~3-5μs (仅更新数据 + 重放)
- GPU 计算: ~100μs
- 总时间: ~103-105μs
- CPU 占比: 3-5%

加速: 1.2-1.4x (取决于 kernel 数量和 GPU 计算时间)
```

**性能提升因素**:
- ✅ **减少 kernel launch 次数**：每次 launch 节省 5-15μs[[1]](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- ✅ **消除 CPU-GPU 同步开销**：通过图内依赖管理
- ✅ **降低驱动层开销**：批量处理 kernel 启动

**注意事项**:
- 性能提升与 kernel 数量正相关（更多 kernels → 更大收益）
- 当 GPU 计算时间占主导（>200μs）时，加速比下降
- vLLM 实测：batched tokens > 200 时，收益递减[[2]](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

**数据来源**:
- [NVIDIA CUDA Graphs Performance Blog (2024)](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [Robust Compiler Support for CUDA Graphs in PyTorch (arXiv 2025)](https://arxiv.org/html/2503.19779v2)

---

## 5. 块表（block_table）的作用是什么？

### 核心概念

**block_table 是逻辑块到物理块的映射表**，是实现 PagedAttention 的关键。

### 图示

```
逻辑序列: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
           └─ Block 0 ─┘ └─ Block 1 ─┘ └─ Block 2 ─┘

block_table = [5, 2, 9]
              ↑   ↑   ↑
              │   │   └─ 逻辑块 2 → 物理块 9
              │   └─────── 逻辑块 1 → 物理块 2
              └──────────── 逻辑块 0 → 物理块 5

物理内存布局:
[Block 0][Block 1][Block 2][Block 3][Block 4][Block 5][Block 6][Block 7][Block 8][Block 9]
                    ↓           ↓                    ↓
                  逻辑块1      逻辑块0              逻辑块2
                  (物理2)      (物理5)              (物理9)
```

### 数据结构

```python
class Sequence:
    block_table = []  # 例如: [5, 2, 9]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        """获取第 i 个逻辑块的 tokens"""
        start = i * self.block_size
        end = min((i + 1) * self.block_size, self.num_tokens)
        return self.token_ids[start:end]
```

### Flash Attention 内部逻辑（简化）

```python
# 伪代码：Flash Attention 如何使用 block_table
def flash_attn_with_kvcache(q, k_cache, v_cache, block_tables):
    batch_size, num_heads, head_dim = q.shape

    for i in range(batch_size):
        seq_len = context_lens[i]
        block_table = block_tables[i]  # [5, 2, 9]

        # 计算需要读取的物理块
        num_blocks = (seq_len + block_size - 1) // block_size
        for j in range(num_blocks):
            physical_block_id = block_table[j]  # ← 逻辑块 → 物理块

            # 从物理块读取 KV
            k_block = k_cache[physical_block_id]
            v_block = v_cache[physical_block_id]

            # 计算 attention
            attn_output += attention(q[i], k_block, v_block)

    return attn_output
```

### 关键优势

1. **内存不连续**: 可以使用分散的物理块
2. **动态分配**: 序列增长时追加新块
3. **共享缓存**: 相同前缀的序列共享 block_table 条目

---

## 6. 抢占（Preemption）何时发生？

### 核心概念

**抢占 = 暂停当前序列，释放其内存，将其他序列移入运行队列**

类似于操作系统的进程调度。

### 触发条件

在 **Decode 阶段**，当一个序列需要追加新 token 但内存不足时触发：

```python
# 调度器逻辑
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()

    # 检查是否有空间追加新 token
    while not self.block_manager.can_append(seq):
        if self.running:
            # 内存不足！抢占最后一个运行的序列
            self.preempt(self.running.pop())
        else:
            # 没有其他序列可抢占，抢占当前序列
            self.preempt(seq)
            break
    else:
        # 有空间，继续运行
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

### 抢占实现

**抢占操作** (`nanovllm/engine/scheduler.py:60-63`):

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING       # 改为等待状态
    self.block_manager.deallocate(seq)        # 释放所有块！
    self.waiting.appendleft(seq)              # 放到等待队列头部
```

**释放块** (`nanovllm/engine/block_manager.py:84-91`):

```python
def deallocate(self, seq: Sequence):
    # 反向遍历，减少引用计数
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)  # 归还到空闲队列

    seq.num_cached_tokens = 0  # 清空缓存标记
    seq.block_table.clear()    # 清空块表
```

### 抢占的代价

```
被抢占的序列:
- 释放所有 KV cache
- 下次 Prefill 需要重新计算
- 除非有 Prefix Caching 可以恢复

优化:
- 抢占最后生成的序列（通常较短）
- Prefix Caching 可以部分缓解重算
```

---

# 使用场景

## 选择 vLLM 的场景

✅ **生产环境部署**
- 需要稳定性和可靠性
- 需要多模型支持
- 需要完整的 API Server

✅ **大规模服务**
- 高并发请求
- 需要自动扩缩容
- 分布式部署

✅ **企业应用**
- 需要监控和日志
- 需要容错机制
- 需要技术支持

✅ **快速集成**
- LangChain/LlamaIndex 集成
- OpenAI API 兼容
- 云平台一键部署

## 选择 Nano-vLLM 的场景

✅ **学习和研究**
- 理解 LLM 推理原理
- 研究优化技术
- 教学和演示

✅ **原型开发**
- 快速验证想法
- 实验新算法
- 定制化需求

✅ **资源受限环境**
- 轻量级部署
- 单机推理
- 特定模型（Qwen）

✅ **代码阅读**
- 整个代码库可读完
- 易于理解和修改
- 适合教学

---

# 学习路径

## 如果你初学 LLM 推理

**推荐：先 Nano-vLLM，后 vLLM**

### 第一阶段：理解核心概念

```
1. 运行 example.py
   └─ 感受推理效果

2. 阅读 scheduler.py
   ├─ 理解两阶段调度
   └─ 理解抢占机制

3. 阅读 block_manager.py
   ├─ 理解 PagedAttention
   ├─ 理解 Prefix Caching
   └─ 理解 block_table

4. 阅读 model_runner.py
   ├─ 理解 Prefill/Decode 流程
   ├─ 理解 CUDA Graph
   └─ 理解张量并行

5. 阅读 attention.py
   └─ 理解 Flash Attention 集成
```

### 第二阶段：深入源码

```
6. 运行 bench.py
   └─ 观察性能指标

7. 修改参数实验
   ├─ 调整 block_size
   ├─ 调整 max_num_seqs
   └─ 观察性能变化

8. 添加简单功能
   └─ 例如：添加新的采样策略

9. 阅读 vLLM 源码对比
   └─ 找到相同概念在 vLLM 中的实现
```

### 第三阶段：生产实践

```
10. 部署 vLLM
    ├─ 安装: pip install vllm
    ├─ 启动 API Server
    └─ 测试性能

11. 监控和优化
    ├─ 配置 Prometheus
    ├─ 观察资源使用
    └─ 优化参数

12. 集成到应用
    ├─ LangChain 集成
    ├─ OpenAI API 兼容
    └─ 云平台部署
```

## 核心概念检查清单

学习后你应该能回答：

- [ ] PagedAttention 如何解决内存碎片？
- [ ] Prefix Caching 如何提升性能？
- [ ] Prefill 和 Decode 的区别是什么？
- [ ] CUDA Graph 如何减少开销？
- [ ] 块表（block_table）的作用是什么？
- [ ] 抢占（preemption）何时发生？
- [ ] vLLM 和 Nano-vLLM 的主要差异是什么？
- [ ] 什么场景应该用 vLLM，什么场景用 Nano-vLLM？

---

## 总结

这 6 个核心概念构成了 LLM 推理引擎的基础：

1. **PagedAttention**: 分页内存管理，提高利用率
2. **Prefix Caching**: 哈希去重，共享计算结果
3. **Prefill/Decode**: 分离处理，针对性优化
4. **CUDA Graph**: 捕获计算图，减少 CPU 开销
5. **block_table**: 逻辑映射到物理，实现分散存储
6. **Preemption**: 动态调度，优先级管理

理解这些概念后，你可以：
- ✅ 修改调度策略
- ✅ 优化内存分配
- ✅ 添加新的优化技术
- ✅ 支持更多模型架构
- ✅ 部署生产级推理服务

---

## 参考资源

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM 论文**: https://arxiv.org/abs/2309.06180
- **Nano-vLLM GitHub**: https://github.com/GeeeekExplorer/nano-vllm


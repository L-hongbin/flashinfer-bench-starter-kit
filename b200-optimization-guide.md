# B200（Blackwell SM100）GPU 通用 CUDA 算子优化指南

面向 **B200（Blackwell Data Center Architecture, SM100）** 的通用 CUDA
算子优化建议。

# 1. 架构参数与硬件特征

## 1.1 基本架构信息

  |参数 |                            B200（Blackwell SM100）|   说明|
  |-----------|-----------|-----------|
  |架构代号 |                       SM100        |            Blackwell Data Center|
  | SM 数量 |                         148       |              单卡 |
  |最大 resident warps / SM        | 64                        |以 warpgroup 组织
  |最大 resident blocks / SM       | 32                        |Blackwell SM block 限制|
  |寄存器总量 / SM                 | 65,536 (32-bit)           | 64K registers
  |每线程寄存器上限                | 255                       
  |L2 Cache                        | \~126 MB                   |全芯片共享
  |Unified L1/Texture/Shared pool  | 256 KB / SM               | 统一池
  |最大 Shared Memory / SM         | 228 KB                    |来自 unified pool carveout
  |默认 Shared / Block              | 48 KB                     |非 opt-in 上限
  |最大 Shared / Block（opt-in）    | \~227 KB                 | 受 SM carveout 限制
  | Memory Bandwidth | ~8 TB/s | HBM3e 显存带宽（全局内存吞吐）|
| Warp Size | 32 threads | CUDA warp 粒度，与 CUDA 规范一致|
| 最大 Threads/Block | 1024 | CUDA 规范上限 |
| 最大 resident Threads/SM | 2048 threads | 由 warp count × warp size 推导 |


------------------------------------------------------------------------

## 1.2 Unified L1 / Shared Carveout 机制

Blackwell 使用统一的 256KB 片上存储池：

Unified Pool (256KB per SM) - Shared Memory carveout (max 228KB) -
L1/Texture cache remainder

Shared memory 增大将压缩 L1 容量。

------------------------------------------------------------------------

# 2. 编译目标与架构路径

## 2.1 推荐编译方式

``` bash
# 通用 Blackwell 目标
nvcc -arch=sm_100 -O3 your_kernel.cu

# 使用架构加速特性（TMA / Warpgroup MMA 等）
nvcc -arch=sm100a -O3 your_kernel.cu
```

建议始终保留 `sm_100` fallback。

------------------------------------------------------------------------

# 3. Blackwell 核心架构变化

## 3.1 Warpgroup 执行模型

在 Blackwell（SM100）中，CUDA 的基本调度单位仍然是：Warp = 32 threads, 但在高吞吐 Tensor Core 路径上，采用 warp-specialized 协作模型：通常 4 个 warps（128 threads）协作执行一个 Tensor Core tile

⚠ Warpgroup 是执行协作概念，不是新的调度单位：

- Occupancy 仍按 warp 计算

- 仍受 64 warps / SM 限制

- 仍受寄存器和 shared memory 限制

主要作用

- 驱动新一代 Tensor Core 指令（如 tcgen05.mma）

- 支持更大的 MMA tile

- 提升 Tensor Core 利用率

- 更易构建 TMA + Compute 多 stage pipeline

设计建议

- GEMM-like kernel 以 128 threads 为基本计算单元

- CTA 中 warp 数量建议为 4 的倍数

- 每个 warpgroup 负责一个 MMA tile

------------------------------------------------------------------------

## 3.2 TMA（Tensor Memory Accelerator）

Blackwell 上推荐使用 TMA 作为主搬运路径。

适用场景：

-   规则 2D / 3D tile

-   2KB 连续块

-   GEMM / 卷积 / block reduction

------------------------------------------------------------------------

# 4. 内存层次优化策略

## 4.1 L2 Cache 利用

Blackwell 的 126MB L2 是核心优势。

优化原则：

-   跨 CTA 重用数据停留在 L2
-   persistent kernel 提高 L2 命中
-   tile 设计考虑 L2 容量

------------------------------------------------------------------------

## 4.2 向量化访问

  |数据类型 |  推荐|
  |-----------|-----------|
  | FP16   |    `half2`|
  | BF16    |   `__nv_bfloat162`|
  |  FP32    |   `float2` / `float4`|

------------------------------------------------------------------------

# 5. Pipeline 设计

推荐使用：

TMA load → Compute → Store multi-stage pipeline。

目标：

-   减少 memory dependency stall
-   提升 tensor core utilization

------------------------------------------------------------------------

# 6. Roofline 驱动优化流程

判断算子属于：

-   Compute-bound
-   L2-bound
-   DRAM-bound
-   Launch-bound

使用 `nsys` + `ncu` 做闭环调优。

------------------------------------------------------------------------

# 7. 总结

Blackwell 优化核心在于：

-   Warpgroup 执行模型
-   TMA 主搬运路径
-   超大 L2 重用策略
-   深度 pipeline 设计

目标是提升 tensor pipe 利用率，减少 HBM 往返。

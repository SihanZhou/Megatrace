# NCCL/CUDA 拦截日志功能使用说明

## 功能概述

本模块实现了对 NCCL API 和 CUDA Kernel 的拦截，并将时间信息记录到文件中。支持通过环境变量控制是否启用拦截以及日志文件路径。

## 环境变量配置

### 1. `NCCL_MEGATRACE_ENABLE`
- **功能**: 控制是否开启拦截和日志记录
- **取值**: `1` 表示启用，其他值或不设置表示禁用
- **示例**: `export NCCL_MEGATRACE_ENABLE=1`

### 2. `NCCL_MEGATRACE_LOG_PATH`
- **功能**: 指定日志文件写入的目录路径
- **默认值**: `./mega_log`（如果未设置）
- **示例**: `export NCCL_MEGATRACE_LOG_PATH=./mega_log`

### 3. Rank 信息获取
模块会自动从以下环境变量中获取 rank 信息（按优先级）：
- `LOCAL_RANK` (优先)
- `RANK`
- `WORLD_RANK`
- 如果都未设置，默认使用 rank 0

## 日志文件格式

每个 rank 会生成独立的日志文件：`{log_path}/rank_{rank}.log`

### 日志文件结构

#### 文件头
```
=== NCCL/CUDA Intercept Log Started ===
Rank: 0
Start Time: 1234567890.123456789
Log File: ./mega_log/rank_0.log
=======================================
```

#### NCCL API 日志格式
```
时间戳,NCCL_API,操作名,count=数量,datatype=类型,stream=地址,comm=地址,group_hash=哈希值
```

**示例**:
```
1703123456.123456789,NCCL_API,AllReduce,count=1024,datatype=4,stream=0x7f8a1c000000,comm=0x7f8a1c001000,group_hash=0xdeadbeef12345678
```

#### CUDA Kernel 日志格式
```
时间戳,CUDA_KERNEL,func=地址,grid=(x,y,z),block=(x,y,z),stream=地址,is_nccl=0/1
```

**示例**:
```
1703123456.234567890,CUDA_KERNEL,func=0x7f8a1c002000,grid=(256,1,1),block=(256,1,1),stream=0x7f8a1c000000,is_nccl=1
```

#### 文件尾
```
=== NCCL/CUDA Intercept Log Ended ===
End Time: 1234567890.987654321
=====================================
```

## 使用方法

### 1. 编译拦截库

```bash
./build_intercept.sh
```

### 2. 设置环境变量并运行程序

```bash
export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./mega_log
export LD_PRELOAD=./libnccl_cuda_intercept.so

torchrun --nproc_per_node 8 test.py
```

### 3. 使用验证脚本

```bash
./verify_intercept.sh
```

验证脚本会自动：
- 设置环境变量
- 运行测试程序
- 检查日志文件是否生成
- 统计拦截的 API 和 Kernel 数量
- 显示日志文件信息

## 日志内容说明

### NCCL API 拦截

拦截的 NCCL API 包括：
- `ncclAllReduce`
- `ncclAllGather`
- `ncclReduceScatter`
- `ncclReduce`
- `ncclBroadcast`
- `ncclSend`
- `ncclRecv`
- `ncclSendRecv`

每条日志记录包含：
- **时间戳**: 精确到纳秒的时间戳
- **操作名**: NCCL 操作类型
- **count**: 数据元素数量
- **datatype**: NCCL 数据类型（整数编码）
- **stream**: CUDA stream 地址
- **comm**: NCCL communicator 地址
- **group_hash**: 通信组的哈希值

### CUDA Kernel 拦截

拦截所有 `cudaLaunchKernel` 调用，每条日志记录包含：
- **时间戳**: 精确到纳秒的时间戳
- **func**: Kernel 函数地址
- **grid**: Grid 维度 (x, y, z)
- **block**: Block 维度 (x, y, z)
- **stream**: CUDA stream 地址
- **is_nccl**: 是否为 NCCL kernel (1=是, 0=否)

### NCCL Kernel 识别

模块通过 `dladdr()` 函数检查 kernel 函数地址所在的共享库来判断是否为 NCCL kernel：
1. 检查库文件名是否包含 "nccl"
2. 检查库基址是否在已加载的 NCCL 库中

## 注意事项

1. **性能影响**: 拦截和日志记录会带来一定的性能开销，建议仅在调试或分析时启用
2. **日志文件大小**: 长时间运行的程序可能产生较大的日志文件，注意磁盘空间
3. **并发安全**: 日志写入是线程安全的，多个线程可以安全地同时写入
4. **Rank 识别**: 确保程序设置了正确的 rank 环境变量，否则所有进程可能写入同一个文件

## 故障排查

### 问题1: 没有生成日志文件

**可能原因**:
- `NCCL_MEGATRACE_ENABLE` 未设置为 `1`
- 日志目录创建失败（权限问题）
- 程序未实际调用 NCCL 或 CUDA 函数

**解决方法**:
- 检查环境变量设置
- 检查目录权限
- 确认程序确实使用了 NCCL/CUDA

### 问题2: 所有 rank 写入同一个文件

**可能原因**:
- 环境变量中未设置 `LOCAL_RANK` 或 `RANK`
- 所有进程获取到的 rank 都是默认值 0

**解决方法**:
- 确保使用 `torchrun` 或正确设置 rank 环境变量
- 检查环境变量是否被正确传递

### 问题3: 无法识别 NCCL kernel

**可能原因**:
- Kernel 是 JIT 编译的（不在共享库中）
- Kernel 函数指针是间接调用
- NCCL 库版本或加载方式特殊

**解决方法**:
- 这是正常现象，某些情况下确实无法识别
- 可以通过 `is_nccl=0` 的 kernel 日志来手动分析

## 示例输出

运行验证脚本后的输出示例：

```
========================================
NCCL/CUDA Intercept Verification Script
========================================

✓ Found libnccl_cuda_intercept.so
✓ Environment variables set:
  NCCL_MEGATRACE_ENABLE=1
  NCCL_MEGATRACE_LOG_PATH=./mega_log
  LD_PRELOAD=./libnccl_cuda_intercept.so

Running test with 2 processes...

Rank 0 log file:
  File: ./mega_log/rank_0.log
  Size: 2048 bytes
  Lines: 15
  NCCL API calls: 3
  CUDA Kernel launches: 12
  NCCL Kernels: 2

Rank 1 log file:
  File: ./mega_log/rank_1.log
  Size: 1987 bytes
  Lines: 14
  NCCL API calls: 3
  CUDA Kernel launches: 11
  NCCL Kernels: 2

========================================
Verification Summary
========================================
Total NCCL API calls across all ranks: 6
Total CUDA Kernel launches: 23
Total NCCL Kernels detected: 4

✓ Intercept is working correctly!
  - NCCL API interception: ✓
  - CUDA Kernel interception: ✓
  - NCCL Kernel detection: ✓
```


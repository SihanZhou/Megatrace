# NCCL Kernel 符号表识别使用说明

## 概述

`nccl_cuda_intercept.cc` 现在支持通过符号表识别 `cudaLaunchKernel` 拦截到的具体 NCCL 操作。这需要：

1. 使用 `gen_nvida_symbols.py` 生成符号表文件
2. 在编译时链接 protobuf 库（可选，如果不使用 protobuf 会回退到原有的 dladdr 方法）
3. 设置环境变量指定符号表文件路径

## 使用步骤

### 方式 A: 自动生成（推荐）

**无需手动执行脚本！** 库会在首次加载时自动生成符号表。

只需要：

1. **确保 `gen_nvida_symbols.py` 脚本可访问**：
   - 脚本应该在与 `libnccl_cuda_intercept.so` 相同的目录
   - 或通过环境变量 `GEN_NVIDIA_SYMBOLS_SCRIPT` 指定路径
   - 或在当前工作目录

2. **确保 Python 可用**：
   - 系统中有 `python3` 或 `python` 命令
   - 或通过环境变量 `PYTHON` 指定 Python 解释器路径

3. **运行程序**（库会自动生成符号表）：
   ```bash
   LD_PRELOAD=./libnccl_cuda_intercept.so your_program
   ```

**自动生成的行为**：
- 如果符号表文件不存在，会自动调用 `gen_nvida_symbols.py` 生成
- 符号表文件默认保存在与 NCCL 库相同的目录，或 `/tmp` 目录
- 可以通过 `NCCL_SYMBOL_FILE` 环境变量指定符号表文件路径
- 可以通过 `NCCL_AUTO_GEN_SYMBOLS=0` 禁用自动生成

### 方式 B: 手动生成（可选）

如果你想手动生成符号表文件：

```bash
# 方法 1: 自动检测 NCCL 库
python3 gen_nvida_symbols.py /path/to/nccl_symbols.pb

# 方法 2: 指定 NCCL 库路径
NCCL_LIB=/usr/local/lib/libnccl.so.2 python3 gen_nvida_symbols.py /path/to/nccl_symbols.pb

# 方法 3: 同时指定 Flash Attention 库（如果需要）
FA_LIB=/path/to/libflash_attn.so NCCL_LIB=/path/to/libnccl.so.2 \
    python3 gen_nvida_symbols.py /path/to/nccl_symbols.pb
```

生成的 `nccl_symbols.pb` 文件包含所有 NCCL kernel 函数的地址偏移量和元信息。

### 步骤 2: 编译拦截库（启用 protobuf 支持）

如果要使用符号表识别功能，需要：

1. **包含 protobuf 头文件**：

   在 `nccl_cuda_intercept.cc` 文件开头添加：
   ```cpp
   // 根据实际情况选择其中一个
   #include "xpu_timer/protos/hook.pb.h"
   // 或
   #include "py_xpu_timer/hook.pb.h"
   ```

2. **取消注释 protobuf 解析代码**：

   在 `load_symbol_table_protobuf()` 函数中，取消注释实际的 protobuf 解析代码（大约在第 580-620 行）。

3. **编译时链接 protobuf 库**：

   ```bash
   g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
       -ldl -lnccl -lcudart -lprotobuf -std=c++11 \
       -I/path/to/protobuf/include \
       -L/path/to/protobuf/lib
   ```

**注意**：如果不启用 protobuf 支持，代码会自动回退到使用 `dladdr()` 方法，只能判断是否是 NCCL kernel，无法识别具体操作类型。

### 步骤 3: 设置环境变量（可选）

```bash
# 指定符号表文件路径（可选，如果不设置会自动选择默认路径）
export NCCL_SYMBOL_FILE=/path/to/nccl_symbols.pb

# 禁用自动生成（如果已手动生成符号表）
export NCCL_AUTO_GEN_SYMBOLS=0

# 指定 Python 解释器（如果 python3 不在 PATH 中）
export PYTHON=/usr/bin/python3

# 指定 gen_nvida_symbols.py 脚本路径（如果不在默认位置）
export GEN_NVIDIA_SYMBOLS_SCRIPT=/path/to/gen_nvida_symbols.py

# 启用日志记录（可选）
export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./mega_log

# 设置 rank（用于日志文件名）
export LOCAL_RANK=0  # 或 RANK=0

# 启用调试输出（可选，可以看到自动生成的详细信息）
export NCCL_INTERCEPT_DEBUG=debug
```

### 步骤 4: 运行程序

```bash
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

**首次运行时会自动生成符号表**，后续运行会直接使用已生成的符号表文件。

## 工作原理

### 自动生成流程

```
库加载时（constructor 函数）
    ↓
检查符号表文件是否存在
    ↓
如果不存在且 NCCL_AUTO_GEN_SYMBOLS != "0"：
    ↓
查找 Python 解释器（python3 或 python）
    ↓
查找 gen_nvida_symbols.py 脚本路径
    ↓
查找 NCCL 库路径（通过 /proc/self/maps 或 dladdr）
    ↓
执行：python3 gen_nvida_symbols.py <symbol_file_path>
    ↓
检查生成的文件是否存在
    ↓
加载符号表
```

### 符号表识别流程

```
cudaLaunchKernel 被调用
    ↓
拦截函数获取 kernel 函数指针
    ↓
计算偏移量 = 函数地址 - 库基址（通过 dladdr）
    ↓
在符号表中查找该偏移量
    ↓
如果找到且 func_type == "NCCL"：
    - 返回 1（是 NCCL kernel）
    - 可以获取详细信息：coll_type, algo, operation, dtype
    ↓
记录日志（包含详细信息）
```

### 回退机制

如果符号表未加载或查找失败，会自动回退到原有的 `dladdr()` 方法：

```
检查库文件名是否包含 "nccl"
    ↓
或检查库基址是否匹配 NCCL 库
    ↓
返回 1 或 0（只能判断是否是 NCCL kernel，无法识别具体操作）
```

## 日志格式

### 启用符号表识别后的日志格式

```
# 基本格式（如果没有符号表或查找失败）
1704067200.123456789,CUDA_KERNEL,func=0x40331b0,grid=(256,1,1),block=(256,1,1),stream=0x7f8a1c000000,is_nccl=1

# 增强格式（如果成功识别到 NCCL 操作）
1704067200.123456789,CUDA_KERNEL,func=0x40331b0,grid=(256,1,1),block=(256,1,1),stream=0x7f8a1c000000,is_nccl=1,coll_type=AllReduce,algo=COLLNET_DIRECT_LL,operation=Sum,dtype=bf16,kernel_name=AllReduce_COLLNET_DIRECT_LL_Sum_bf16
```

### 字段说明

- `coll_type`: 集合通信类型（AllReduce, AllGather, ReduceScatter, Broadcast, SendRecv, Reduce）
- `algo`: 算法（COLLNET_DIRECT_LL, RING, TREE 等）
- `operation`: 归约操作（Sum, Prod, Max, Min）
- `dtype`: 数据类型（fp16, bf16, fp32, fp64, int8, int32, int64 等）
- `kernel_name`: 完整的 kernel 函数名

## 故障排查

### 问题 1: 自动生成失败

**症状**：调试日志显示 "Failed to generate symbol table" 或 "Python not found"

**可能原因**：
1. Python 解释器未找到
2. `gen_nvida_symbols.py` 脚本未找到
3. NCCL 库路径未找到
4. 脚本执行失败（依赖缺失等）

**解决方法**：
1. 检查 Python：`which python3` 或 `which python`
2. 设置 Python 路径：`export PYTHON=/usr/bin/python3`
3. 检查脚本路径：`ls -l gen_nvida_symbols.py`
4. 设置脚本路径：`export GEN_NVIDIA_SYMBOLS_SCRIPT=/path/to/gen_nvida_symbols.py`
5. 查看详细日志：设置 `NCCL_INTERCEPT_DEBUG=debug` 查看自动生成的命令和错误信息
6. 手动生成：如果自动生成失败，可以手动执行脚本生成符号表

### 问题 2: 符号表未加载

**症状**：日志中只有 `is_nccl=1`，没有详细信息

**可能原因**：
1. 自动生成失败且未手动生成
2. `NCCL_SYMBOL_FILE` 环境变量指向的文件不存在
3. protobuf 支持未启用
4. 符号表文件格式不正确

**解决方法**：
1. 检查环境变量：`echo $NCCL_SYMBOL_FILE`
2. 检查文件是否存在：`ls -l $NCCL_SYMBOL_FILE` 或检查默认路径
3. 查看调试输出：设置 `NCCL_INTERCEPT_DEBUG=debug` 查看详细日志
4. 手动生成符号表：参考"方式 B: 手动生成"

### 问题 2: 偏移量不匹配

**症状**：符号表已加载但无法识别 kernel

**可能原因**：
1. NCCL 库版本不同（生成符号表时和运行时使用的库版本不一致）
2. 库被重新编译
3. ASLR（地址空间布局随机化）影响（通常不影响偏移量）

**解决方法**：
1. 确保使用相同的 NCCL 库版本
2. 重新生成符号表文件

### 问题 3: Protobuf 编译错误

**症状**：编译时找不到 protobuf 头文件或链接错误

**解决方法**：
1. 安装 protobuf 开发库：`apt-get install libprotobuf-dev` 或类似
2. 检查包含路径和库路径是否正确
3. 如果不想使用 protobuf，可以不启用，代码会自动回退到 dladdr 方法

## 性能考虑

1. **符号表加载**：只在初始化时执行一次，开销可忽略
2. **查找操作**：使用 `std::unordered_map`，O(1) 时间复杂度
3. **线程安全**：使用互斥锁保护，但查找操作很快，影响很小
4. **内存占用**：符号表通常很小（几 KB 到几十 KB）

## 扩展功能

### 不使用 Protobuf 的替代方案

如果不想依赖 protobuf，可以：

1. **修改 `gen_nvida_symbols.py` 输出 JSON 格式**：
   ```python
   import json
   # ... 生成符号信息 ...
   with open(sys.argv[1], "w") as f:
       json.dump(symbol_dict, f)
   ```

2. **在 `load_symbol_table_protobuf()` 中实现 JSON 解析**：
   ```cpp
   #include <nlohmann/json.hpp>  // 或其他 JSON 库
   // 或使用简单的文本格式解析
   ```

3. **使用简单的文本格式**（每行一个映射）：
   ```
   offset:func_name:func_type:coll_type:algo:operation:dtype
   ```

## 示例

### 示例 1: 最简单的使用（自动生成）

```bash
# 1. 编译（启用 protobuf）
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -ldl -lnccl -lcudart -lprotobuf -std=c++11

# 2. 确保 gen_nvida_symbols.py 在同一目录或可访问

# 3. 运行程序（会自动生成符号表）
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

### 示例 2: 完整配置（手动指定路径）

```bash
# 1. 编译（启用 protobuf）
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -ldl -lnccl -lcudart -lprotobuf -std=c++11

# 2. 设置环境变量
export NCCL_SYMBOL_FILE=./nccl_symbols.pb
export GEN_NVIDIA_SYMBOLS_SCRIPT=./gen_nvida_symbols.py
export PYTHON=python3
export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./mega_log
export LOCAL_RANK=0
export NCCL_INTERCEPT_DEBUG=debug  # 查看自动生成的详细信息

# 3. 运行程序（会自动生成符号表）
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

### 示例 3: 手动生成后使用

```bash
# 1. 手动生成符号表
python3 gen_nvida_symbols.py ./nccl_symbols.pb

# 2. 编译（启用 protobuf）
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -ldl -lnccl -lcudart -lprotobuf -std=c++11

# 3. 设置环境变量（禁用自动生成）
export NCCL_SYMBOL_FILE=./nccl_symbols.pb
export NCCL_AUTO_GEN_SYMBOLS=0
export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./mega_log
export LOCAL_RANK=0

# 4. 运行程序
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

### 查看日志

```bash
# 查看日志文件
cat ./mega_log/rank_0.log

# 过滤 NCCL kernel
grep "is_nccl=1" ./mega_log/rank_0.log

# 查看特定操作（如 AllReduce）
grep "coll_type=AllReduce" ./mega_log/rank_0.log
```


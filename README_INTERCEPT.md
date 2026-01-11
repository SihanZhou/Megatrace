# NCCL和CUDA Kernel拦截模块

这是一个独立的拦截模块，用于拦截NCCL函数调用和CUDA kernel启动。

## 功能特性

1. **NCCL函数拦截**
   - 拦截所有主要的NCCL集合通信函数
   - 支持：AllReduce, AllGather, ReduceScatter, Reduce, Broadcast, Send, Recv, SendRecv
   - 自动建立comm和commId的映射关系

2. **CUDA Kernel拦截**
   - 拦截`cudaLaunchKernel`调用
   - 记录kernel启动信息（函数指针、网格/块维度、流等）

3. **NCCL解析功能**
   - comm和commId的双向映射
   - 通信组哈希值计算
   - devComm提取（可自定义）

## 文件说明

- `nccl_cuda_intercept.h` - 头文件，定义API接口
- `nccl_cuda_intercept.cc` - 实现文件，包含所有拦截逻辑
- `nccl_cuda_intercept_example.cc` - 使用示例

## 编译方法

### 基本编译

```bash
# 编译为共享库
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -ldl -lnccl -lcudart -std=c++11
```

### 如果找不到CUDA库

如果遇到 `cannot find -lcudart` 错误，需要指定CUDA库路径：

```bash
# 方法1：使用环境变量（推荐）
export CUDA_HOME=/usr/local/cuda  # 或你的CUDA安装路径
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -ldl -lnccl -lcudart -std=c++11

# 方法2：直接指定路径
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -ldl -lnccl -lcudart -std=c++11

# 方法3：使用pkg-config（如果已安装）
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    $(pkg-config --cflags --libs cudart) \
    -ldl -lnccl -std=c++11
```

### 如果需要包含comm.h来正确提取devComm

如果项目中有 `include/comm.h` 文件（NCCL内部头文件），可以包含它来正确提取devComm：

```bash
# 如果有include目录
g++ -shared -fPIC -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    -I./include \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -ldl -lnccl -lcudart -std=c++11

# 或者使用自定义提取函数（推荐，不依赖NCCL内部头文件）
# 见 nccl_cuda_intercept_example.cc 中的示例
```

### 查找CUDA安装路径

如果不知道CUDA安装路径，可以尝试：

```bash
# 查找cudart库
find /usr -name "libcudart.so*" 2>/dev/null
find /usr/local -name "libcudart.so*" 2>/dev/null

# 或使用ldconfig
ldconfig -p | grep cudart
```

## 使用方法

### 方法1：使用回调函数（推荐）

```c
#include "nccl_cuda_intercept.h"

// 定义回调函数
void on_nccl_op(const char* op_name, size_t count, ncclDataType_t datatype,
                cudaStream_t stream, ncclComm_t comm, uint64_t group_hash,
                const char* timestamp) {
    // 处理NCCL操作事件
    printf("NCCL %s: count=%zu\n", op_name, count);
}

void on_cuda_kernel(const void* func, dim3 grid_dim, dim3 block_dim,
                    cudaStream_t stream, int is_nccl_kernel, const char* timestamp) {
    // 处理CUDA kernel事件
    printf("CUDA kernel: func=%p\n", func);
}

// 初始化
nccl_cuda_intercept_init(on_nccl_op, on_cuda_kernel);
```

### 方法2：使用LD_PRELOAD

```bash
# 编译为共享库后，使用LD_PRELOAD加载
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

注意：使用LD_PRELOAD时，需要在库加载时自动初始化。可以添加构造函数：

```c
__attribute__((constructor))
void auto_init() {
    // 设置默认回调或使用环境变量配置
    nccl_cuda_intercept_init(nullptr, nullptr);
}
```

## API说明

### 初始化

```c
int nccl_cuda_intercept_init(
    nccl_op_callback_t nccl_cb,      // NCCL操作回调（可为NULL）
    cuda_kernel_callback_t kernel_cb  // CUDA kernel回调（可为NULL）
);
```

### 清理

```c
void nccl_cuda_intercept_cleanup(void);
```

### 启用/禁用

```c
void nccl_cuda_intercept_set_enabled(int enable);
```

### NCCL解析功能

```c
// 获取comm对应的commId
int nccl_get_comm_id(ncclComm_t comm, ncclUniqueId* comm_id);

// 根据commId获取comm
ncclComm_t nccl_get_comm_by_id(const ncclUniqueId* comm_id);

// 计算通信组哈希值
uint64_t nccl_get_group_hash(ncclComm_t comm);

// 提取devComm地址
void* nccl_extract_dev_comm(ncclComm_t comm);

// 设置自定义devComm提取函数
void nccl_set_dev_comm_extractor(void* (*extractor)(ncclComm_t));
```

## 注意事项

1. **devComm提取**：默认实现可能无法正确提取devComm，建议：
   - 包含`include/comm.h`并设置自定义提取函数
   - 或通过offsetof计算偏移量
   - 或使用环境变量配置偏移量

2. **线程安全**：所有函数都是线程安全的

3. **性能影响**：拦截会带来一定的性能开销，建议在需要时启用

4. **NCCL版本兼容性**：不同版本的NCCL可能有不同的内部结构，需要相应调整

## 与原始代码的区别

1. **独立性**：不依赖ring_log、log等模块
2. **回调机制**：使用回调函数而非直接写入日志
3. **可配置性**：支持自定义devComm提取函数
4. **简化**：移除了不必要的依赖，保持代码简洁

## 扩展功能

如果需要识别NCCL kernel，可以：

1. 加载NCCL符号表（参考`gen_nccl_symbols_simple.py`）
2. 在`cudaLaunchKernel`回调中检查函数指针
3. 实现`is_nccl_kernel`函数

示例：

```c
static bool is_nccl_kernel(const void* func) {
    // 加载符号表并检查
    // 返回true如果是NCCL kernel
    return false;
}
```


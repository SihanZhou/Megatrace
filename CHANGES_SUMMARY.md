# 修改总结

## 问题
`gen_nvida_symbols.py` 脚本依赖 `xpu_timer` 或 `py_xpu_timer` 模块，但这些模块在当前项目中不存在，导致自动生成符号表失败。

## 解决方案

### 1. 修改 `gen_nvida_symbols.py` 脚本

**主要改动**：
- ✅ 移除了对 `xpu_timer.protos.hook_pb2` 和 `py_xpu_timer.hook_pb2` 的强制依赖
- ✅ 改为使用 JSON 格式作为默认输出格式（无需额外依赖）
- ✅ 保留了 protobuf 格式支持（如果模块可用）
- ✅ 移除了对 `torch.utils.cpp_extension` 的强制依赖（改为可选）
- ✅ 改进了错误处理和异常捕获

**新的使用方式**：
```bash
# 生成 JSON 格式（推荐，无需依赖）
python3 gen_nvida_symbols.py nccl_symbols.json

# 如果 protobuf 可用，也可以生成 .pb 格式
python3 gen_nvida_symbols.py nccl_symbols.pb
```

### 2. 更新 C++ 代码支持 JSON 格式

**主要改动**：
- ✅ 添加了 `<regex>` 头文件支持
- ✅ 实现了 JSON 格式的符号表加载（`load_symbol_table_protobuf` 函数）
- ✅ 更新了自动生成函数，默认使用 JSON 格式（`.json` 扩展名）
- ✅ 保留了 protobuf 格式支持（如果可用）

**JSON 格式示例**：
```json
{
  "1073741872": {
    "func_name": "AllReduce_COLLNET_DIRECT_LL_Sum_bf16",
    "func_type": "NCCL",
    "coll_type": "AllReduce",
    "algo": "COLLNET_DIRECT_LL",
    "operation": "Sum",
    "dtype": "bf16"
  }
}
```

### 3. 自动生成功能

**默认行为**：
- 自动生成时默认使用 JSON 格式（`.json` 扩展名）
- 如果用户指定 `.pb` 扩展名，脚本会尝试生成 protobuf 格式（如果可用）
- 如果 protobuf 不可用，脚本会自动回退到 JSON 格式

## 依赖关系

### 必需依赖（Python 脚本）
- ✅ Python 3（标准库）
- ✅ `json` 模块（Python 标准库）
- ✅ `subprocess` 模块（Python 标准库）
- ✅ `re` 模块（Python 标准库）

### 可选依赖（Python 脚本）
- ⚪ `torch`（如果可用，用于查找 libtorch_cuda.so）
- ⚪ `flash_attn`（如果可用，用于解析 Flash Attention kernel）
- ⚪ `xpu_timer` 或 `py_xpu_timer`（如果可用，用于生成 protobuf 格式）

### C++ 代码依赖
- ✅ C++11 标准库（`<regex>`, `<unordered_map>`, `<fstream>` 等）
- ⚪ protobuf 库（可选，如果使用 `.pb` 格式）

## 使用示例

### 方式 1: 自动生成（推荐）

```bash
# 直接运行程序，库会自动生成 JSON 格式的符号表
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

### 方式 2: 手动生成

```bash
# 生成 JSON 格式
python3 gen_nvida_symbols.py nccl_symbols.json

# 设置环境变量
export NCCL_SYMBOL_FILE=./nccl_symbols.json

# 运行程序
LD_PRELOAD=./libnccl_cuda_intercept.so your_program
```

## 注意事项

1. **JSON 解析**：当前实现的 JSON 解析使用简单的字符串匹配和正则表达式，对于复杂的 JSON 结构可能不够健壮。如果遇到问题，可以考虑：
   - 使用更简单的 JSON 格式（避免嵌套）
   - 或者引入轻量级 JSON 库（如 `nlohmann/json`）

2. **文件扩展名**：
   - `.json` - JSON 格式（推荐，无需依赖）
   - `.pb` - Protobuf 格式（需要 protobuf 库支持）

3. **向后兼容**：如果项目中已有 protobuf 支持，可以继续使用 `.pb` 格式。

## 待完善的功能

如果 JSON 解析遇到问题，可以考虑以下改进：

1. **使用标准 JSON 库**（需要添加依赖）：
   ```cpp
   #include <nlohmann/json.hpp>
   // 或
   #include <json/json.h>  // jsoncpp
   ```

2. **使用更简单的文本格式**（每行一个映射）：
   ```
   offset:func_name:func_type:coll_type:algo:operation:dtype
   ```

3. **二进制格式**（自定义格式，无需外部库）

## 测试建议

1. 测试自动生成功能：
   ```bash
   export NCCL_INTERCEPT_DEBUG=debug
   LD_PRELOAD=./libnccl_cuda_intercept.so your_program
   ```

2. 检查生成的符号表文件：
   ```bash
   cat nccl_symbols.json | head -20
   ```

3. 验证符号表加载：
   - 查看调试日志，确认符号表已加载
   - 检查日志中是否包含详细的 NCCL 操作信息


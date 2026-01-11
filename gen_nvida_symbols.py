#!/usr/bin/env python3
# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# flake8: noqa: E501,E722,F841,E401
import os
import re
import subprocess
import sys
import json

# 移除对 torch 的依赖，改为直接查找库
try:
    from torch.utils import cpp_extension
    libtorch_cuda = f"{cpp_extension.TORCH_LIB_PATH}/libtorch_cuda.so"
except Exception:
    # 如果 torch 不可用，使用默认路径
    libtorch_cuda = "/usr/local/lib/libtorch_cuda.so"

# 移除对 flash_attn 的依赖（可选功能）
try:
    import flash_attn
    has_flash_attn = True
except:
    has_flash_attn = False

flash_attn_dtype_mapping = {"half_t": "fp16", "bfloat16_t": "bf16"}
nccl_dtype_mapping = {
    "half": "fp16",
    "__nv_bfloat16": "bf16",
    "double": "fp64",
    "float": "fp32",
    "int8_t": "int8",
    "int32_t": "int32",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "u8": "uint8",
    "u32": "uint32",
    "f16": "fp16",
    "f32": "fp32",
    "f64": "fp64",
    "u64": "uint64",
    "bf16": "bf16",
}

dtype_mapping = {**flash_attn_dtype_mapping, **nccl_dtype_mapping}


def list_loaded_libraries_linux():
    """查找已加载的 NCCL 和 Flash Attention 库路径"""
    nccl_lib = None
    flash_attn_lib = None
    try:
        with open("/proc/self/maps", "r") as f:
            lines = f.readlines()
        for line in lines:
            if has_flash_attn and "flash_attn_2_cuda" in line:
                flash_attn_lib = line.strip().split(" ")[-1]
            elif "libnccl.so" in line:
                nccl_lib = line.strip().split(" ")[-1]
    except Exception:
        pass
    
    if nccl_lib is None:
        nccl_lib = libtorch_cuda
    return flash_attn_lib, nccl_lib


def pipe_commands(commands):
    """执行管道命令"""
    if len(commands) == 1:
        result = subprocess.Popen(commands[0], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        return stdout.decode().strip() if stdout else "", stderr.decode().strip() if stderr else ""

    cur = None
    for command in commands:
        cur = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=cur.stdout if cur is not None else None,
        )
    stdout, stderr = cur.communicate()
    return stdout.decode().strip() if stdout else "", stderr.decode().strip() if stderr else ""


def nccl_prser(nccl_lib):
    """解析 NCCL 库的符号表，返回解析器函数"""
    
    def parse_2_22_3(_):
        patterns = [
            re.compile(
                r"([0-9a-f]+) \w ncclDevKernel_(AllReduce|ReduceScatter|SendRecv|Reduce)_(Sum|Prod|Max|Min)_([a-z0-9_]+)_([A-Z_]+)\(ncclDevKernelArgsStorage<4096ul>\)"
            ),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_SendRecv\(ncclDevKernelArgsStorage<4096ul>\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_AllGather_([A-Z_]+)\(ncclDevKernelArgsStorage<4096ul>\)"),
        ]

        def inner(sym, symbols_dict):
            for index, pattern in enumerate(patterns):
                addr = coll_type = operation = dtype = algo = None
                search = pattern.search(sym)
                if search is not None:
                    if index == 0:
                        addr, coll_type, operation, dtype, algo = search.groups()
                        kernel_name = "_".join([coll_type, algo, operation, dtype])
                    elif index == 1:
                        coll_type = "SendRecv"
                        (addr,) = search.groups()
                        kernel_name = "SendRecv"
                    elif index == 2:
                        coll_type = "AllGather"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"

                    addr = int(addr, 16)
                    symbol_info = {
                        "func_name": kernel_name,
                        "func_type": "NCCL",
                        "coll_type": coll_type or "",
                        "algo": algo or "",
                        "operation": operation or "",
                        "dtype": dtype_mapping.get(dtype, dtype) if dtype else "",
                    }
                    symbols_dict[addr] = symbol_info

        return inner

    def parse_2_21_5(_):
        patterns = [
            re.compile(
                r"([0-9a-f]+) \w ncclDevKernel_(AllReduce|ReduceScatter|SendRecv|Reduce)_(Sum|Prod|Max|Min)_([a-z0-9_]+)_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
            ),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_SendRecv\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_AllGather_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
            re.compile(r"([0-9a-f]+) \w ncclDevKernel_Broadcast_([A-Z_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"),
        ]

        def inner(sym, symbols_dict):
            for index, pattern in enumerate(patterns):
                addr = coll_type = operation = dtype = algo = None
                search = pattern.search(sym)
                if search is not None:
                    if index == 0:
                        addr, coll_type, operation, dtype, algo = search.groups()
                        kernel_name = "_".join([coll_type, algo, operation, dtype])
                    elif index == 1:
                        coll_type = "SendRecv"
                        (addr,) = search.groups()
                        kernel_name = "SendRecv"
                    elif index == 2:
                        coll_type = "AllGather"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"
                    elif index == 3:
                        coll_type = "Broadcast"
                        addr, algo = search.groups()
                        kernel_name = f"{coll_type}_{algo}"

                    addr = int(addr, 16)
                    symbol_info = {
                        "func_name": kernel_name,
                        "func_type": "NCCL",
                        "coll_type": coll_type or "",
                        "algo": algo or "",
                        "operation": operation or "",
                        "dtype": dtype_mapping.get(dtype, dtype) if dtype else "",
                    }
                    symbols_dict[addr] = symbol_info

        return inner

    def parse_2_18_5(nccl_lib):
        pattern = re.compile(
            r"([0-9a-f]+) \w ncclKernel_(Broadcast|AllReduce|ReduceScatter|AllGather|SendRecv|Reduce)_([A-Z_]+)_(Sum|Prod|Max|Min)_([a-z0-9_]+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        )

        def inner(sym, symbols_dict):
            search = pattern.search(sym)
            if search is not None:
                addr, coll_type, algo, operation, dtype = search.groups()
                kernel_name = "_".join([coll_type, algo, operation, dtype])
                dtype = dtype_mapping.get(dtype, dtype)

                addr = int(addr, 16)
                symbol_info = {
                    "func_name": kernel_name,
                    "func_type": "NCCL",
                    "coll_type": coll_type,
                    "algo": algo,
                    "operation": operation,
                    "dtype": dtype,
                }
                symbols_dict[addr] = symbol_info

        return inner

    nccl_regex = {
        "NCCL version 2.22.3": [
            "ncclDevKernel_",
            parse_2_22_3,
        ],
        "NCCL version 2.21.5": [
            "ncclDevKernel_",
            parse_2_21_5,
        ],
        "NCCL version 2.18.5": [
            "ncclKernel_",
            parse_2_18_5,
        ],
        "NCCL version 2.20.5": [
            "ncclDevKernel_",
            parse_2_21_5,
        ],
    }

    try:
        nccl_version_command = [
            ["strings", nccl_lib],
            ["grep", "-m1", "-P", "NCCL version.*cuda"],
            ["awk", "-F+", "{print $1}"],
        ]
        nccl_version, _ = pipe_commands(nccl_version_command)
        if nccl_version in nccl_regex:
            kernel_pattern, parser = nccl_regex[nccl_version]
            return kernel_pattern, parser(nccl_lib)
        else:
            # 默认使用 2.21.5 的解析器
            print(f"Warning: Unknown NCCL version '{nccl_version}', using default parser", file=sys.stderr)
            return "ncclDevKernel_", parse_2_21_5(nccl_lib)
    except Exception as e:
        print(f"Error detecting NCCL version: {e}, using default parser", file=sys.stderr)
        return "ncclDevKernel_", parse_2_21_5(nccl_lib)


def search_flash_attn(flash_attn_lib, symbols_dict):
    """搜索 Flash Attention kernel 符号（可选功能）"""
    if not has_flash_attn or not flash_attn_lib:
        return
    
    try:
        flash_attn_dtype = re.compile(r"cutlass::([a-zA-Z0-9_]+)")
        fwd_pattern = re.compile("flash_fwd[a-zA-Z0-9_]+kernel")
        bwd_pattern = re.compile("flash_bwd[a-zA-Z0-9_]+kernel")
        bwd_commands = [
            ["nm", flash_attn_lib],
            ["c++filt"],
            ["grep", "-Po", ".*flash_bwd.*_kernel.*"],
        ]
        fwd_commands = [
            ["nm", flash_attn_lib],
            ["c++filt"],
            ["grep", "-Po", ".*flash_fwd.*_kernel.*"],
        ]
        fwd_symbols, _ = pipe_commands(fwd_commands)
        bwd_symbols, _ = pipe_commands(bwd_commands)
        fwd_symbols = fwd_symbols.split("\n") if fwd_symbols else []
        bwd_symbols = bwd_symbols.split("\n") if bwd_symbols else []

        def parse_syms(syms, operation, pattern):
            for sym in syms:
                if not sym.strip():
                    continue
                try:
                    dtype_match = flash_attn_dtype.search(sym)
                    if not dtype_match:
                        continue
                    dtype = dtype_match.group(1)
                    kernel_name_match = pattern.search(sym)
                    if kernel_name_match is None:
                        continue
                    kernel_name = kernel_name_match.group()
                    addr_str = sym.split()[0]
                    addr = int(addr_str, 16)
                    
                    symbol_info = {
                        "func_name": kernel_name,
                        "func_type": "FA",
                        "dtype": dtype_mapping.get(dtype, dtype),
                        "only_trace": operation == "FaBwd" and "parallel" not in kernel_name,
                        "algo": "",
                        "coll_type": "",
                        "operation": operation,
                    }
                    symbols_dict[addr] = symbol_info
                except Exception as e:
                    print(f"Warning: Failed to parse symbol '{sym}': {e}", file=sys.stderr)
                    continue

        parse_syms(fwd_symbols, "FaFwd", fwd_pattern)
        parse_syms(bwd_symbols, "FaBwd", bwd_pattern)
    except Exception as e:
        print(f"Warning: Flash Attention search failed: {e}", file=sys.stderr)


def search_nccl(nccl_lib, symbols_dict):
    """搜索 NCCL kernel 符号"""
    if not nccl_lib:
        return
    
    try:
        kernel_pattern, parser = nccl_prser(nccl_lib)
        nccl_commands = [
            ["nm", nccl_lib],
            ["grep", kernel_pattern],
            ["grep", "-vP", "__device_stub__"],
            ["c++filt"],
        ]
        nccl_symbols, _ = pipe_commands(nccl_commands)
        nccl_symbols = nccl_symbols.split("\n") if nccl_symbols else []
        
        for sym in nccl_symbols:
            if sym.strip():
                try:
                    parser(sym, symbols_dict)
                except Exception as e:
                    print(f"Warning: Failed to parse symbol '{sym}': {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error searching NCCL symbols: {e}", file=sys.stderr)


def main():
    """主函数：生成符号表文件"""
    if len(sys.argv) < 2:
        print("Usage: gen_nvida_symbols.py <output_file>", file=sys.stderr)
        print("  Output file can be .json (JSON format) or .pb (protobuf format, if protobuf available)", file=sys.stderr)
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # 查找库
    flash_attn_lib, nccl_lib = list_loaded_libraries_linux()
    flash_attn_lib = os.environ.get("FA_LIB", None) or flash_attn_lib
    nccl_lib = os.environ.get("NCCL_LIB", None) or nccl_lib
    
    # 符号字典（使用普通字典替代 protobuf）
    symbols_dict = {}
    
    # 搜索符号
    if flash_attn_lib:
        search_flash_attn(flash_attn_lib, symbols_dict)
    if nccl_lib:
        search_nccl(nccl_lib, symbols_dict)
    
    # 根据文件扩展名选择输出格式
    if output_file.endswith('.json'):
        # JSON 格式（推荐，无需额外依赖）
        with open(output_file, "w") as f:
            json.dump(symbols_dict, f, indent=2)
        print(f"Generated {len(symbols_dict)} symbols in JSON format: {output_file}")
    elif output_file.endswith('.pb'):
        # Protobuf 格式（如果可用）
        try:
            # 尝试导入 protobuf（如果可用）
            try:
                from xpu_timer.protos.hook_pb2 import InterceptSymbolByOffset
            except ImportError:
                try:
                    from py_xpu_timer.hook_pb2 import InterceptSymbolByOffset
                except ImportError:
                    print("Error: Protobuf support not available.", file=sys.stderr)
                    print("Please install xpu_timer or py_xpu_timer, or use JSON format (.json)", file=sys.stderr)
                    sys.exit(1)
            
            addr_to_name = InterceptSymbolByOffset()
            for addr, symbol_info in symbols_dict.items():
                symbol_pb = addr_to_name.symbols[addr]
                symbol_pb.func_name = symbol_info.get("func_name", "")
                symbol_pb.func_type = symbol_info.get("func_type", "")
                symbol_pb.coll_type = symbol_info.get("coll_type", "")
                symbol_pb.algo = symbol_info.get("algo", "")
                symbol_pb.operation = symbol_info.get("operation", "")
                symbol_pb.dtype = symbol_info.get("dtype", "")
            
            with open(output_file, "wb") as f:
                f.write(addr_to_name.SerializeToString())
            print(f"Generated {len(symbols_dict)} symbols in protobuf format: {output_file}")
        except Exception as e:
            print(f"Error generating protobuf file: {e}", file=sys.stderr)
            print("Falling back to JSON format...", file=sys.stderr)
            json_file = output_file.replace('.pb', '.json')
            with open(json_file, "w") as f:
                json.dump(symbols_dict, f, indent=2)
            print(f"Generated {len(symbols_dict)} symbols in JSON format: {json_file}")
    else:
        # 默认使用 JSON 格式
        json_file = output_file + ".json"
        with open(json_file, "w") as f:
            json.dump(symbols_dict, f, indent=2)
        print(f"Generated {len(symbols_dict)} symbols in JSON format: {json_file}")


if __name__ == "__main__":
    main()

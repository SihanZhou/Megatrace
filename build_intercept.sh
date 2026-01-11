#!/bin/bash
# 编译NCCL和CUDA拦截模块的脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building NCCL and CUDA intercept module...${NC}"

# 查找CUDA路径
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -d "$CUDA_HOME" ]; then
    # 尝试从常见路径查找
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_HOME="/opt/cuda"
    else
        # 尝试通过ldconfig查找
        CUDA_LIB=$(ldconfig -p | grep libcudart.so | head -1 | awk '{print $NF}' | xargs dirname | xargs dirname)
        if [ -n "$CUDA_LIB" ]; then
            CUDA_HOME="$CUDA_LIB"
        else
            echo -e "${YELLOW}Warning: CUDA_HOME not found, trying default paths...${NC}"
            CUDA_HOME="/usr/local/cuda"
        fi
    fi
fi

echo -e "${GREEN}Using CUDA_HOME: $CUDA_HOME${NC}"

# 检查CUDA库是否存在
if [ ! -f "$CUDA_HOME/lib64/libcudart.so" ] && [ ! -f "$CUDA_HOME/lib/libcudart.so" ]; then
    echo -e "${RED}Error: Cannot find libcudart.so in $CUDA_HOME${NC}"
    echo -e "${YELLOW}Please set CUDA_HOME environment variable to your CUDA installation path${NC}"
    exit 1
fi

# 设置库路径
if [ -d "$CUDA_HOME/lib64" ]; then
    CUDA_LIB_DIR="$CUDA_HOME/lib64"
elif [ -d "$CUDA_HOME/lib" ]; then
    CUDA_LIB_DIR="$CUDA_HOME/lib"
else
    echo -e "${RED}Error: Cannot find CUDA lib directory${NC}"
    exit 1
fi

# 编译选项
CXX_FLAGS="-shared -fPIC -std=c++11"
INCLUDE_DIRS="-I./include"
LIB_DIRS="-L$CUDA_LIB_DIR"
LIBS="-ldl -lnccl -lcudart"

# 检查是否包含comm.h
if [ -f "./include/comm.h" ]; then
    echo -e "${GREEN}Found comm.h, including it in build${NC}"
    INCLUDE_DIRS="$INCLUDE_DIRS -I./include"
fi

# 编译
echo -e "${GREEN}Compiling...${NC}"
g++ $CXX_FLAGS -o libnccl_cuda_intercept.so nccl_cuda_intercept.cc \
    $INCLUDE_DIRS \
    $LIB_DIRS \
    $LIBS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Output: libnccl_cuda_intercept.so${NC}"
    
    # 显示库依赖
    echo -e "${GREEN}Library dependencies:${NC}"
    ldd libnccl_cuda_intercept.so | grep -E "(cudart|nccl)" || true
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi


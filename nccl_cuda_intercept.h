#ifndef NCCL_CUDA_INTERCEPT_H
#define NCCL_CUDA_INTERCEPT_H

#include <nccl.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 回调函数类型定义
// ============================================================================

/**
 * NCCL操作回调函数
 * @param op_name 操作名称（如 "AllReduce", "AllGather"）
 * @param count 数据元素数量
 * @param datatype NCCL数据类型
 * @param stream CUDA流
 * @param comm NCCL通信器
 * @param group_hash 通信组哈希值
 * @param timestamp 时间戳（秒.纳秒格式）
 */
typedef void (*nccl_op_callback_t)(
    const char* op_name,
    size_t count,
    ncclDataType_t datatype,
    cudaStream_t stream,
    ncclComm_t comm,
    uint64_t group_hash,
    const char* timestamp
);

/**
 * CUDA Kernel启动回调函数
 * @param func 内核函数指针
 * @param grid_dim 网格维度
 * @param block_dim 块维度
 * @param stream CUDA流
 * @param is_nccl_kernel 是否为NCCL内核
 * @param timestamp 时间戳（秒.纳秒格式）
 */
typedef void (*cuda_kernel_callback_t)(
    const void* func,
    dim3 grid_dim,
    dim3 block_dim,
    cudaStream_t stream,
    int is_nccl_kernel,
    const char* timestamp
);

// ============================================================================
// 初始化与配置
// ============================================================================

/**
 * 初始化拦截模块
 * @param nccl_cb NCCL操作回调函数（可为NULL）
 * @param kernel_cb CUDA kernel回调函数（可为NULL）
 * @return 0表示成功，非0表示失败
 */
int nccl_cuda_intercept_init(
    nccl_op_callback_t nccl_cb,
    cuda_kernel_callback_t kernel_cb
);

/**
 * 清理拦截模块
 */
void nccl_cuda_intercept_cleanup(void);

/**
 * 启用/禁用拦截
 * @param enable 1启用，0禁用
 */
void nccl_cuda_intercept_set_enabled(int enable);

// ============================================================================
// NCCL解析功能
// ============================================================================

/**
 * 根据comm获取commId
 * @param comm NCCL通信器
 * @param comm_id 输出的commId
 * @return 0表示成功，非0表示失败
 */
int nccl_get_comm_id(ncclComm_t comm, ncclUniqueId* comm_id);

/**
 * 根据commId获取comm
 * @param comm_id NCCL唯一ID
 * @return NCCL通信器，失败返回NULL
 */
ncclComm_t nccl_get_comm_by_id(const ncclUniqueId* comm_id);

/**
 * 计算通信组的哈希值
 * @param comm NCCL通信器
 * @return 哈希值，失败返回0
 */
uint64_t nccl_get_group_hash(ncclComm_t comm);

/**
 * 从comm中提取devComm地址
 * @param comm NCCL通信器
 * @return devComm地址，失败返回NULL
 * 
 * 注意：此函数依赖于NCCL内部结构，可能需要根据NCCL版本调整
 * 如果无法正确提取，可以设置自定义提取函数
 */
void* nccl_extract_dev_comm(ncclComm_t comm);

/**
 * 设置自定义devComm提取函数
 * @param extractor 提取函数，参数为ncclComm_t，返回devComm地址
 */
void nccl_set_dev_comm_extractor(void* (*extractor)(ncclComm_t));

// ============================================================================
// NCCL信息查询
// ============================================================================

/**
 * NCCL操作信息结构
 */
typedef struct {
    size_t count;
    ncclDataType_t datatype;
    ncclComm_t comm;
    char operation_type[32];  // "AllReduce", "AllGather", etc.
} nccl_op_info_t;

/**
 * 获取NCCL操作信息（从devComm）
 * @param dev_comm devComm地址
 * @param info 输出的操作信息
 * @return 0表示成功，非0表示失败
 */
int nccl_get_op_info(void* dev_comm, nccl_op_info_t* info);

#ifdef __cplusplus
}
#endif

#endif // NCCL_CUDA_INTERCEPT_H


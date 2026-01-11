/**
 * 独立的NCCL和CUDA Kernel拦截模块
 * 
 * 功能：
 * 1. NCCL函数拦截（AllReduce, AllGather, ReduceScatter等）
 * 2. CUDA Kernel拦截（cudaLaunchKernel）
 * 3. NCCL解析功能（comm映射、groupHash计算）
 * 
 * 使用方式：
 * 1. 调用 nccl_cuda_intercept_init() 初始化
 * 2. 设置回调函数接收拦截事件
 * 3. 程序退出时调用 nccl_cuda_intercept_cleanup() 清理
 */

#include "nccl_cuda_intercept.h"
#include <dlfcn.h>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <stack>
#include <queue>
#include <string>
#include <memory>
#include <vector>
#include <regex>
#include <time.h>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// ============================================================================
// 前向声明和类型定义
// ============================================================================

// NCCL函数指针类型
typedef ncclResult_t (*ncclAllReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclAllGather_t)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduceScatter_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclBroadcast_t)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSend_t)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclRecv_t)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSendRecv_t)(const void*, size_t, ncclDataType_t, int, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclCommInitRank_t)(ncclComm_t*, int, ncclUniqueId, int);
typedef ncclResult_t (*ncclCommInitRankConfig_t)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
typedef ncclResult_t (*ncclCommInitAll_t)(ncclComm_t*, int, const int*);
typedef ncclResult_t (*ncclGetUniqueId_t)(ncclUniqueId*);

// CUDA函数指针类型
typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

// ============================================================================
// 全局变量
// ============================================================================

// 真实函数指针
static ncclAllReduce_t real_ncclAllReduce = nullptr;
static ncclAllGather_t real_ncclAllGather = nullptr;
static ncclReduceScatter_t real_ncclReduceScatter = nullptr;
static ncclReduce_t real_ncclReduce = nullptr;
static ncclBroadcast_t real_ncclBroadcast = nullptr;
static ncclSend_t real_ncclSend = nullptr;
static ncclRecv_t real_ncclRecv = nullptr;
static ncclSendRecv_t real_ncclSendRecv = nullptr;
static ncclCommInitRank_t real_ncclCommInitRank = nullptr;
static ncclCommInitRankConfig_t real_ncclCommInitRankConfig = nullptr;
static ncclCommInitAll_t real_ncclCommInitAll = nullptr;
static ncclGetUniqueId_t real_ncclGetUniqueId = nullptr;
static cudaLaunchKernel_t real_cudaLaunchKernel = nullptr;

// 回调函数
static nccl_op_callback_t g_nccl_callback = nullptr;
static cuda_kernel_callback_t g_kernel_callback = nullptr;
static int g_intercept_enabled = 1;

// 线程安全
static std::mutex g_dlsym_mutex;
static std::mutex g_comm_mapping_mutex;
static std::mutex g_nccl_info_mutex;

// NCCL 库句柄
static void* g_nccl_handle = nullptr;
static std::mutex g_nccl_handle_mutex;
static void* g_self_base = nullptr;  // 我们自己的库基址，用于递归检测

// NCCL comm映射
struct NcclUniqueIdHash {
    std::size_t operator()(const ncclUniqueId& id) const {
        uint64_t h = 0xdeadbeef;
        const char* ptr = reinterpret_cast<const char*>(&id);
        for (size_t i = 0; i < sizeof(ncclUniqueId); i++) {
            h ^= h >> 32;
            h *= 0x8db3db47fa2994adULL;
            h += ptr[i];
        }
        return static_cast<std::size_t>(h);
    }
};

struct NcclUniqueIdEqual {
    bool operator()(const ncclUniqueId& lhs, const ncclUniqueId& rhs) const {
        return memcmp(&lhs, &rhs, sizeof(ncclUniqueId)) == 0;
    }
};

static std::unordered_map<ncclUniqueId, ncclComm_t, NcclUniqueIdHash, NcclUniqueIdEqual> g_comm_id_to_comm;
static std::unordered_map<ncclComm_t, ncclUniqueId, std::hash<void*>, std::equal_to<void*>> g_comm_to_comm_id;
static std::mutex g_unique_id_stack_mutex;
static std::stack<ncclUniqueId> g_unique_id_stack;

// NCCL操作信息存储
struct NcclInfo {
    size_t count;
    ncclComm_t comm;
    ncclDataType_t datatype;
    std::string operation_type;
};

static std::unordered_map<void*, std::queue<NcclInfo>> g_nccl_info_map;

// Stream操作计数
static std::mutex g_stream_opcount_mutex;
static std::unordered_map<cudaStream_t, int64_t> g_stream_to_opcount;

// 自定义devComm提取函数
static void* (*g_dev_comm_extractor)(ncclComm_t) = nullptr;

// ============================================================================
// NCCL Kernel 符号表识别相关
// ============================================================================

// NCCL Kernel 符号信息结构
struct NcclKernelSymbol {
    std::string func_name;
    std::string func_type;    // "NCCL", "FA", etc.
    std::string coll_type;   // "AllReduce", "AllGather", "ReduceScatter", etc.
    std::string algo;        // "COLLNET_DIRECT_LL", "RING", "TREE", etc.
    std::string operation;   // "Sum", "Prod", "Max", "Min"
    std::string dtype;       // "fp16", "bf16", "fp32", "fp64", etc.
};

// 偏移量 -> 符号信息的映射表
static std::unordered_map<int64_t, NcclKernelSymbol> g_offset_to_symbol;
static std::mutex g_symbol_mutex;
static bool g_symbols_loaded = false;
static bool g_symbols_load_attempted = false;  // 是否已尝试加载（避免重复尝试）

// ============================================================================
// 文件日志记录相关
// ============================================================================

// 日志文件
static std::ofstream g_log_file;
static std::mutex g_log_file_mutex;
static int g_log_enabled = 0;  // 是否启用日志（由环境变量控制）
static std::string g_log_path;  // 日志文件路径
static int g_rank = -1;  // 当前rank（从环境变量获取）
static int g_debug_enabled = 0;  // 是否启用调试输出（由NCCL_INTERCEPT_DEBUG环境变量控制）

// 调试输出宏（仅在调试模式启用时输出）
#define DEBUG_LOG(...) \
    do { \
        if (g_debug_enabled) { \
            fprintf(stderr, "[NCCL_INTERCEPT] " __VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 初始化：获取我们自己的库基址
 */
static void init_self_base() {
    if (g_self_base) {
        return;
    }
    
    Dl_info info;
    // 使用一个已知在我们库中的函数来获取基址
    if (dladdr((void*)nccl_cuda_intercept_init, &info) != 0) {
        g_self_base = info.dli_fbase;
    }
}

/**
 * 打开真实的 NCCL 库
 */
static void* get_nccl_handle() {
    if (g_nccl_handle) {
        return g_nccl_handle;
    }
    
    std::lock_guard<std::mutex> lock(g_nccl_handle_mutex);
    if (g_nccl_handle) {
        return g_nccl_handle;
    }
    
    // 初始化我们自己的库基址
    init_self_base();
    
    // 尝试打开真实的 NCCL 库
    // 使用 RTLD_LAZY | RTLD_LOCAL 避免符号冲突
    const char* nccl_libs[] = {
        "libnccl.so.2",
        "libnccl.so",
        "libnccl.so.2.27",
        "libnccl.so.2.26",
        "libnccl.so.2.25",
        nullptr
    };
    
    for (int i = 0; nccl_libs[i] != nullptr; i++) {
        dlerror();  // 清除之前的错误
        void* handle = dlopen(nccl_libs[i], RTLD_LAZY | RTLD_LOCAL);
        if (handle) {
            g_nccl_handle = handle;
            return handle;
        }
    }
    
    // 如果都失败了，返回 nullptr 表示失败
    // 不要使用 RTLD_NEXT，因为可能导致递归
    return nullptr;
}

/**
 * 线程安全的dlsym，从真实的 NCCL 库中获取符号
 */
template<typename T>
static T safe_dlsym(const char* symbol, T* cached_ptr) {
    if (*cached_ptr) {
        return *cached_ptr;
    }
    
    std::lock_guard<std::mutex> lock(g_dlsym_mutex);
    if (*cached_ptr) {
        return *cached_ptr;
    }
    
    // 清除之前的错误
    dlerror();
    
    // 首先尝试从真实的 NCCL 库中获取
    void* nccl_handle = get_nccl_handle();
    void* ptr = nullptr;
    
    if (nccl_handle) {
        // 从打开的 NCCL 库中获取符号
        ptr = dlsym(nccl_handle, symbol);
    }
    
    if (!ptr) {
        // 如果 dlopen 失败或符号未找到，尝试 RTLD_NEXT
        // 但需要确保不会递归（通过检查符号地址是否在我们自己的库中）
        dlerror();  // 清除之前的错误
        ptr = dlsym(RTLD_NEXT, symbol);
        
        // 检查是否找到了我们自己的函数（递归检测）
        if (ptr && g_self_base) {
            Dl_info info;
            if (dladdr(ptr, &info) != 0) {
                // 如果函数在我们自己的库中，这是递归！
                if (info.dli_fbase == g_self_base) {
                    ptr = nullptr;
                }
            }
        }
    }
    
    if (!ptr) {
        // 记录错误但不输出（避免干扰正常流程）
        const char* error = dlerror();
        (void)error;  // 避免未使用变量警告
        return nullptr;
    }
    
    *cached_ptr = reinterpret_cast<T>(ptr);
    return *cached_ptr;
}

/**
 * 计算哈希值
 */
static uint64_t hash_bytes(const void* bytes, size_t size) {
    uint64_t h = 0xdeadbeef;
    const char* ptr = static_cast<const char*>(bytes);
    for (size_t i = 0; i < size; i++) {
        h ^= h >> 32;
        h *= 0x8db3db47fa2994adULL;
        h += ptr[i];
    }
    return h;
}

/**
 * 哈希ncclUniqueId
 */
static uint64_t hash_unique_id(const ncclUniqueId& id) {
    return hash_bytes(&id, sizeof(ncclUniqueId));
}

/**
 * 获取stream的操作计数
 */
static int64_t next_opcount_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_stream_opcount_mutex);
    int64_t& counter = g_stream_to_opcount[stream];
    return ++counter;
}

/**
 * 获取时间戳字符串
 */
static void get_timestamp_string(char* buf, size_t buf_size) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(buf, buf_size, "%ld.%09ld", ts.tv_sec, ts.tv_nsec);
}

/**
 * 从环境变量获取rank信息
 * 优先尝试 LOCAL_RANK，如果没有则尝试 RANK
 */
static int get_rank_from_env() {
    const char* local_rank_str = getenv("LOCAL_RANK");
    if (local_rank_str) {
        return atoi(local_rank_str);
    }
    
    const char* rank_str = getenv("RANK");
    if (rank_str) {
        return atoi(rank_str);
    }
    
    // 如果都没有，尝试从 WORLD_RANK 获取
    const char* world_rank_str = getenv("WORLD_RANK");
    if (world_rank_str) {
        return atoi(world_rank_str);
    }
    
    return 0;  // 默认返回0
}

// 日志文件初始化标志（延迟初始化）
static int g_log_file_initialized = 0;

/**
 * 延迟初始化日志文件（在第一次写入时调用）
 * 避免在构造函数中使用C++对象导致段错误
 */
static void ensure_log_file_initialized() {
    if (g_log_file_initialized) {
        return;
    }
    
    // 使用静态局部变量确保只初始化一次
    static int init_guard = 0;
    if (init_guard) {
        return;
    }
    init_guard = 1;
    
    try {
        // 获取日志路径
        const char* log_path_str = getenv("NCCL_MEGATRACE_LOG_PATH");
        const char* log_path = nullptr;
        if (!log_path_str || strlen(log_path_str) == 0) {
            log_path = "./mega_log";  // 默认路径
        } else {
            log_path = log_path_str;
        }
        
        // 保存路径到g_log_path（现在C++运行时应该已经初始化）
        g_log_path = log_path;
        
        // 获取rank信息
        if (g_rank < 0) {
            g_rank = get_rank_from_env();
        }
        
        // 创建日志目录（如果不存在）
        struct stat st;
        if (stat(log_path, &st) != 0) {
            char cmd[512];
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", log_path);
            int ret = system(cmd);
            (void)ret;  // 忽略返回值
        }
        
        // 构建日志文件名
        char log_filename[512];
        snprintf(log_filename, sizeof(log_filename), "%s/rank_%d.log", log_path, g_rank);
        
        // 打开日志文件
        std::lock_guard<std::mutex> lock(g_log_file_mutex);
        g_log_file.open(log_filename, std::ios::out | std::ios::app);
        if (!g_log_file.is_open()) {
            DEBUG_LOG("Error: Failed to open log file: %s\n", log_filename);
            g_log_enabled = 0;
            return;
        }
        
        // 写入日志头
        char timestamp[64];
        get_timestamp_string(timestamp, sizeof(timestamp));
        g_log_file << "=== NCCL/CUDA Intercept Log Started ===" << std::endl;
        g_log_file << "Rank: " << g_rank << std::endl;
        g_log_file << "Start Time: " << timestamp << std::endl;
        g_log_file << "Log File: " << log_filename << std::endl;
        g_log_file << "=======================================" << std::endl;
        g_log_file.flush();
        
        g_log_file_initialized = 1;
        DEBUG_LOG("Log file opened: %s (rank %d)\n", log_filename, g_rank);
        
        // 输出到stderr确保可见（即使DEBUG未启用）
        fprintf(stderr, "[NCCL_INTERCEPT] Log file opened: %s (rank %d)\n", log_filename, g_rank);
        fflush(stderr);
    } catch (...) {
        // 如果出现异常，禁用日志
        g_log_enabled = 0;
        DEBUG_LOG("Exception in ensure_log_file_initialized, disabling logging\n");
    }
}

/**
 * 初始化日志系统
 * 读取环境变量 NCCL_MEGATRACE_ENABLE 和 NCCL_MEGATRACE_LOG_PATH
 * 注意：此函数在构造函数中调用，只设置标志，不打开文件
 */
static void init_log_system() {
    // 检查是否启用日志
    const char* enable_str = getenv("NCCL_MEGATRACE_ENABLE");
    if (!enable_str || strcmp(enable_str, "1") != 0) {
        g_log_enabled = 0;
        return;
    }
    
    g_log_enabled = 1;
    
    // 获取rank信息（只使用基本C函数）
    g_rank = get_rank_from_env();
    
    // 不在这里打开文件，延迟到第一次写入时
    // 这样可以避免在构造函数中使用C++对象
}

/**
 * 写入NCCL操作日志
 */
static void write_nccl_log(const char* op_name, size_t count, ncclDataType_t datatype,
                           cudaStream_t stream, ncclComm_t comm, uint64_t group_hash,
                           const char* timestamp) {
    if (!g_log_enabled) return;
    
    // 延迟初始化日志文件（在第一次写入时）
    ensure_log_file_initialized();
    
    if (!g_log_file_initialized) return;
    
    std::lock_guard<std::mutex> lock(g_log_file_mutex);
    if (!g_log_file.is_open()) return;
    
    // 格式化日志：时间戳,类型,操作名,count,datatype,stream,comm,group_hash
    g_log_file << timestamp << ",NCCL_API," << op_name
               << ",count=" << count
               << ",datatype=" << static_cast<int>(datatype)
               << ",stream=" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
               << ",comm=" << std::hex << reinterpret_cast<uintptr_t>(comm) << std::dec
               << ",group_hash=0x" << std::hex << group_hash << std::dec
               << std::endl;
    g_log_file.flush();
}

// 前向声明：获取 NCCL kernel 详细信息
static int get_nccl_kernel_info(const void* func, NcclKernelSymbol* symbol);

/**
 * 写入CUDA Kernel日志（增强版，支持 NCCL 操作详细信息）
 */
static void write_kernel_log(const void* func, dim3 grid_dim, dim3 block_dim,
                             cudaStream_t stream, int is_nccl_kernel, const char* timestamp) {
    if (!g_log_enabled) return;
    
    // 延迟初始化日志文件（在第一次写入时）
    ensure_log_file_initialized();
    
    if (!g_log_file_initialized) return;
    
    std::lock_guard<std::mutex> lock(g_log_file_mutex);
    if (!g_log_file.is_open()) return;
    
    // 基本日志格式：时间戳,类型,func,grid,block,stream,is_nccl
    g_log_file << timestamp << ",CUDA_KERNEL"
               << ",func=" << std::hex << reinterpret_cast<uintptr_t>(func) << std::dec
               << ",grid=(" << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << ")"
               << ",block=(" << block_dim.x << "," << block_dim.y << "," << block_dim.z << ")"
               << ",stream=" << std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
               << ",is_nccl=" << (is_nccl_kernel ? "1" : "0");
    
    // 如果是 NCCL kernel，尝试获取详细信息
    if (is_nccl_kernel) {
        NcclKernelSymbol symbol;
        if (get_nccl_kernel_info(func, &symbol)) {
            // 添加详细信息：操作类型、算法、归约操作、数据类型
            g_log_file << ",coll_type=" << symbol.coll_type;
            if (!symbol.algo.empty()) {
                g_log_file << ",algo=" << symbol.algo;
            }
            if (!symbol.operation.empty()) {
                g_log_file << ",operation=" << symbol.operation;
            }
            if (!symbol.dtype.empty()) {
                g_log_file << ",dtype=" << symbol.dtype;
            }
            if (!symbol.func_name.empty()) {
                g_log_file << ",kernel_name=" << symbol.func_name;
            }
        }
    }
    
    g_log_file << std::endl;
    g_log_file.flush();
}

/**
 * 关闭日志文件
 */
static void close_log_file() {
    if (!g_log_enabled) return;
    
    std::lock_guard<std::mutex> lock(g_log_file_mutex);
    if (g_log_file.is_open()) {
        char timestamp[64];
        get_timestamp_string(timestamp, sizeof(timestamp));
        g_log_file << "=== NCCL/CUDA Intercept Log Ended ===" << std::endl;
        g_log_file << "End Time: " << timestamp << std::endl;
        g_log_file << "=====================================" << std::endl;
        g_log_file.close();
    }
    g_log_enabled = 0;
}

/**
 * 从ncclComm中提取devComm
 * 注意：这依赖于NCCL内部结构
 */
static void* extract_dev_comm(ncclComm_t comm) {
    if (!comm) return nullptr;
    
    // 如果设置了自定义提取函数，使用它
    if (g_dev_comm_extractor) {
        return g_dev_comm_extractor(comm);
    }
    
    // 默认实现：尝试通过结构体访问
    // 注意：这需要知道NCCL内部结构，可能不适用于所有版本
    // 为了保持独立性，这里使用一个简化的方法
    
    // 方法：假设可以通过类型转换访问（需要包含comm.h）
    // 如果无法包含comm.h，可以：
    // 1. 使用offsetof计算偏移量
    // 2. 通过环境变量配置偏移量
    // 3. 使用自定义提取函数
    
    // 临时方案：返回comm本身（用户应设置自定义提取函数）
    return comm;
}

/**
 * 计算函数指针相对于库基址的偏移量
 */
static int64_t get_function_offset(const void* func) {
    Dl_info info;
    if (dladdr(func, &info) == 0) {
        return -1;  // 无法获取库信息
    }
    
    // 计算偏移量 = 函数地址 - 库基址
    return (char*)func - (char*)info.dli_fbase;
}

/**
 * 从 JSON 或 protobuf 文件加载符号表
 * 
 * 支持格式：
 * 1. JSON 格式（推荐，无需额外依赖）：文件扩展名为 .json
 * 2. Protobuf 格式（可选）：文件扩展名为 .pb
 * 
 * JSON 格式示例：
 * {
 *   "1073741872": {
 *     "func_name": "AllReduce_COLLNET_DIRECT_LL_Sum_bf16",
 *     "func_type": "NCCL",
 *     "coll_type": "AllReduce",
 *     "algo": "COLLNET_DIRECT_LL",
 *     "operation": "Sum",
 *     "dtype": "bf16"
 *   }
 * }
 */
static bool load_symbol_table_protobuf(const std::string& file_path) {
    int current_rank = get_rank_from_env();
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] load_symbol_table_protobuf called with path: %s\n", 
            current_rank, file_path.c_str());
    fflush(stderr);
    
    // 检查文件是否存在
    std::ifstream file(file_path);
    if (!file.is_open()) {
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Failed to open file: %s\n", current_rank, file_path.c_str());
        fflush(stderr);
        return false;
    }
    
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] File opened successfully, checking size...\n", current_rank);
    fflush(stderr);
    
    // 检查文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] File size: %zu bytes\n", current_rank, file_size);
    fflush(stderr);
    
    if (file_size == 0 || file_size > 10 * 1024 * 1024) {  // 最大 10MB
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] File size invalid (0 or >10MB), closing\n", current_rank);
        fflush(stderr);
        file.close();
        return false;
    }
    
    // 根据文件扩展名选择解析方式
    bool is_json = file_path.size() > 5 && file_path.substr(file_path.size() - 5) == ".json";
    bool is_pb = file_path.size() > 3 && file_path.substr(file_path.size() - 3) == ".pb";
    
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] File format: is_json=%d, is_pb=%d\n", 
            current_rank, is_json ? 1 : 0, is_pb ? 1 : 0);
    fflush(stderr);
    
    if (is_json) {
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Entering JSON parsing branch...\n", current_rank);
        fflush(stderr);
        // ========================================================================
        // JSON 格式解析（推荐，无需额外依赖）
        // ========================================================================
        try {
            struct timespec parse_start, parse_end;
            clock_gettime(CLOCK_REALTIME, &parse_start);
            
            int current_rank = get_rank_from_env();
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Reading JSON file (size: %zu bytes)...\n", 
                    current_rank, file_size);
            fflush(stderr);
            
            // 读取 JSON 内容
            std::string json_content((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
            file.close();
            
            clock_gettime(CLOCK_REALTIME, &parse_end);
            double read_elapsed = (parse_end.tv_sec - parse_start.tv_sec) + (parse_end.tv_nsec - parse_start.tv_nsec) / 1e9;
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] File read completed (%.2f seconds), starting parsing...\n", 
                    current_rank, read_elapsed);
            fflush(stderr);
            
            // 简单的 JSON 解析（不依赖外部库）
            // 格式：{"offset": {"func_name": "...", "func_type": "...", ...}, ...}
            // 注意：调用者已经持有 g_symbol_mutex 锁，这里不需要再次获取
            g_offset_to_symbol.clear();
            
            // 使用简单的字符串解析（避免引入 JSON 库依赖）
            // 查找所有 "offset": { ... } 模式
            size_t pos = 0;
            size_t total_keys = 0;
            size_t parsed_count = 0;
            size_t loop_iterations = 0;
            
            clock_gettime(CLOCK_REALTIME, &parse_start);
            
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Starting JSON parsing loop, content size: %zu bytes\n", 
                    current_rank, json_content.size());
            fflush(stderr);
            
            // 辅助函数：从对象字符串中提取字段值（使用简单的字符串查找，比正则快很多）
            auto extract_field_fast = [](const std::string& obj_str, const char* field_name) -> std::string {
                std::string pattern = std::string("\"") + field_name + "\":\"";
                size_t field_pos = obj_str.find(pattern);
                if (field_pos == std::string::npos) {
                    // 尝试带空格的格式: "field": "value"
                    pattern = std::string("\"") + field_name + "\": \"";
                    field_pos = obj_str.find(pattern);
                }
                if (field_pos == std::string::npos) {
                    return "";
                }
                size_t value_start = field_pos + pattern.length();
                size_t value_end = obj_str.find("\"", value_start);
                if (value_end == std::string::npos) {
                    return "";
                }
                return obj_str.substr(value_start, value_end - value_start);
            };
            
            // 先检查 JSON 格式：应该以 { 开头
            if (json_content.empty() || json_content[0] != '{') {
                fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] ERROR: Invalid JSON format (does not start with '{')\n", current_rank);
                fflush(stderr);
                return false;
            }
            
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] JSON format check passed, entering main loop...\n", current_rank);
            fflush(stderr);
            
            while ((pos = json_content.find("\"", pos)) != std::string::npos) {
                loop_iterations++;
                
                // 前10次迭代打印详细信息
                if (loop_iterations <= 10) {
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Loop iteration %zu, pos=%zu\n", 
                            current_rank, loop_iterations, pos);
                    fflush(stderr);
                }
                size_t key_start = pos + 1;
                size_t key_end = json_content.find("\"", key_start);
                if (key_end == std::string::npos) break;
                
                // 快速检查是否为数字键（避免创建临时字符串）
                bool is_numeric = true;
                for (size_t i = key_start; i < key_end; i++) {
                    if (json_content[i] < '0' || json_content[i] > '9') {
                        is_numeric = false;
                        break;
                    }
                }
                
                if (!is_numeric) {
                    pos = key_end + 1;
                    continue;
                }
                
                total_keys++;
                
                // 前10个符号打印详细信息
                if (total_keys <= 10) {
                    std::string key_str = json_content.substr(key_start, key_end - key_start);
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Found numeric key #%zu: '%s' at pos %zu\n", 
                            current_rank, total_keys, key_str.c_str(), key_start);
                    fflush(stderr);
                }
                
                // 每解析100个符号打印一次进度（更频繁的进度更新）
                if (total_keys % 100 == 0) {
                    struct timespec now;
                    clock_gettime(CLOCK_REALTIME, &now);
                    double elapsed = (now.tv_sec - parse_start.tv_sec) + (now.tv_nsec - parse_start.tv_nsec) / 1e9;
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Parsing progress: %zu symbols processed (%.2f seconds, %zu loop iterations)...\n", 
                            current_rank, total_keys, elapsed, loop_iterations);
                    fflush(stderr);
                }
                
                // 解析偏移量（直接使用字符串，避免创建临时对象）
                int64_t offset = 0;
                const char* key_ptr = json_content.c_str() + key_start;
                size_t key_len = key_end - key_start;
                for (size_t i = 0; i < key_len; i++) {
                    offset = offset * 10 + (key_ptr[i] - '0');
                }
                
                // 查找对应的值对象
                size_t obj_start = json_content.find("{", key_end);
                if (obj_start == std::string::npos) {
                    if (total_keys <= 10) {
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] No '{' found after key #%zu, breaking\n", 
                                current_rank, total_keys);
                        fflush(stderr);
                    }
                    break;
                }
                
                // 找到匹配的右括号
                int brace_count = 1;
                size_t obj_end = obj_start + 1;
                size_t brace_search_iterations = 0;
                while (brace_count > 0 && obj_end < json_content.size()) {
                    brace_search_iterations++;
                    if (brace_search_iterations > 1000000) {  // 防止无限循环
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] ERROR: Brace matching loop exceeded limit for key #%zu\n", 
                                current_rank, total_keys);
                        fflush(stderr);
                        break;
                    }
                    if (json_content[obj_end] == '{') brace_count++;
                    else if (json_content[obj_end] == '}') brace_count--;
                    obj_end++;
                }
                
                if (brace_count == 0) {
                    // 使用字符串视图避免复制（C++17），否则使用子串
                    std::string obj_str = json_content.substr(obj_start, obj_end - obj_start);
                    
                    if (total_keys <= 3) {
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Extracting fields for key #%zu, obj_str size: %zu\n", 
                                current_rank, total_keys, obj_str.size());
                        fflush(stderr);
                    }
                    
                    // 解析对象中的字段（使用快速字符串查找）
                    NcclKernelSymbol symbol;
                    symbol.func_name = extract_field_fast(obj_str, "func_name");
                    symbol.func_type = extract_field_fast(obj_str, "func_type");
                    symbol.coll_type = extract_field_fast(obj_str, "coll_type");
                    symbol.algo = extract_field_fast(obj_str, "algo");
                    symbol.operation = extract_field_fast(obj_str, "operation");
                    symbol.dtype = extract_field_fast(obj_str, "dtype");
                    
                    if (total_keys <= 3) {
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Extracted func_type='%s' for key #%zu\n", 
                                current_rank, symbol.func_type.c_str(), total_keys);
                        fflush(stderr);
                    }
                    
                    if (!symbol.func_type.empty()) {
                        g_offset_to_symbol[offset] = symbol;
                        parsed_count++;
                    }
                } else {
                    if (total_keys <= 10) {
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] WARNING: Brace count mismatch (%d) for key #%zu\n", 
                                current_rank, brace_count, total_keys);
                        fflush(stderr);
                    }
                }
                
                pos = obj_end;
            }
            
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Main loop completed: %zu loop iterations, %zu numeric keys found, %zu symbols parsed\n", 
                    current_rank, loop_iterations, total_keys, parsed_count);
            fflush(stderr);
            
            clock_gettime(CLOCK_REALTIME, &parse_end);
            double parse_elapsed = (parse_end.tv_sec - parse_start.tv_sec) + (parse_end.tv_nsec - parse_start.tv_nsec) / 1e9;
            
            if (!g_offset_to_symbol.empty()) {
                DEBUG_LOG("Loaded %zu symbols from JSON file: %s\n", 
                          g_offset_to_symbol.size(), file_path.c_str());
                fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Parsing completed: %zu symbols loaded (parse time: %.2f seconds)\n", 
                        current_rank, g_offset_to_symbol.size(), parse_elapsed);
                fflush(stderr);
                return true;
            } else {
                fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Warning: No symbols loaded from JSON file\n", current_rank);
                fflush(stderr);
            }
        } catch (const std::exception& e) {
            int current_rank = get_rank_from_env();
            DEBUG_LOG("Exception while parsing JSON: %s\n", e.what());
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Exception while parsing JSON: %s\n", 
                    current_rank, e.what());
            fflush(stderr);
            return false;
        }
    } else if (is_pb) {
        // ========================================================================
        // Protobuf 格式解析（可选，需要 protobuf 库）
        // ========================================================================
        file.close();
        
        // 重新以二进制模式打开
        std::ifstream pb_file(file_path, std::ios::binary);
        if (!pb_file.is_open()) {
            return false;
        }
        
        std::vector<char> buffer(file_size);
        pb_file.read(buffer.data(), file_size);
        pb_file.close();
        
        // TODO: 如果需要 protobuf 支持，取消注释以下代码并包含相应的头文件
        /*
        try {
            // 包含 protobuf 头文件（根据实际情况调整路径）
            // #include "xpu_timer/protos/hook.pb.h"
            // 或
            // #include "py_xpu_timer/hook.pb.h"
            
            InterceptSymbolByOffset intercepted_symbols;
            if (!intercepted_symbols.ParseFromArray(buffer.data(), file_size)) {
                DEBUG_LOG("Failed to parse protobuf from %s\n", file_path.c_str());
                return false;
            }
            
            std::lock_guard<std::mutex> lock(g_symbol_mutex);
            g_offset_to_symbol.clear();
            
            for (const auto& entry : intercepted_symbols.symbols()) {
                int64_t offset = entry.first;
                const auto& symbol_pb = entry.second;
                
                NcclKernelSymbol symbol;
                symbol.func_name = symbol_pb.func_name();
                symbol.func_type = symbol_pb.func_type();
                symbol.coll_type = symbol_pb.coll_type();
                symbol.algo = symbol_pb.algo();
                symbol.operation = symbol_pb.operation();
                symbol.dtype = symbol_pb.dtype();
                
                g_offset_to_symbol[offset] = symbol;
            }
            
            return true;
        } catch (const std::exception& e) {
            DEBUG_LOG("Exception while parsing protobuf: %s\n", e.what());
            return false;
        }
        */
        
        DEBUG_LOG("Protobuf format detected but protobuf support not enabled. Use JSON format instead.\n");
        return false;
    } else {
        // 未知格式，尝试作为 JSON 解析
        file.close();
        std::string json_path = file_path + ".json";
        return load_symbol_table_protobuf(json_path);
    }
    
    return false;
}

/**
 * 查找 NCCL 库路径
 */
static std::string find_nccl_library_path() {
    // 方法1: 通过 /proc/self/maps 查找
    std::ifstream maps_file("/proc/self/maps");
    if (maps_file.is_open()) {
        std::string line;
        while (std::getline(maps_file, line)) {
            if (line.find("libnccl.so") != std::string::npos) {
                size_t pos = line.find_last_of('/');
                if (pos != std::string::npos) {
                    std::string lib_path = line.substr(pos + 1);
                    // 提取完整路径（从行尾提取）
                    size_t path_start = line.find_last_of(' ');
                    if (path_start != std::string::npos) {
                        std::string full_path = line.substr(path_start + 1);
                        maps_file.close();
                        return full_path;
                    }
                }
            }
        }
        maps_file.close();
    }
    
    // 方法2: 通过 dladdr 查找
    void* nccl_handle = get_nccl_handle();
    if (nccl_handle) {
        Dl_info info;
        // 尝试通过 dlsym 获取一个 NCCL 函数来定位库
        void* test_func = dlsym(nccl_handle, "ncclGetVersion");
        if (test_func && dladdr(test_func, &info) != 0 && info.dli_fname) {
            return std::string(info.dli_fname);
        }
    }
    
    return "";
}

/**
 * 查找 gen_nvida_symbols.py 脚本路径
 */
static std::string find_gen_symbols_script() {
    // 方法1: 从环境变量获取
    const char* script_path = getenv("GEN_NVIDIA_SYMBOLS_SCRIPT");
    if (script_path && strlen(script_path) > 0) {
        return std::string(script_path);
    }
    
    // 方法2: 尝试常见路径
    // 假设脚本在与 so 库相同的目录或父目录
    Dl_info info;
    if (dladdr((void*)nccl_cuda_intercept_init, &info) != 0 && info.dli_fname) {
        std::string so_path = info.dli_fname;
        size_t last_slash = so_path.find_last_of('/');
        if (last_slash != std::string::npos) {
            std::string dir = so_path.substr(0, last_slash);
            // 尝试几个可能的路径
            std::vector<std::string> candidates = {
                dir + "/gen_nvida_symbols.py",
                dir + "/../gen_nvida_symbols.py",
                dir + "/../../gen_nvida_symbols.py",
                "./gen_nvida_symbols.py",
                "gen_nvida_symbols.py"
            };
            
            for (const auto& candidate : candidates) {
                std::ifstream test_file(candidate);
                if (test_file.good()) {
                    test_file.close();
                    return candidate;
                }
            }
        }
    }
    
    // 方法3: 尝试当前工作目录
    std::ifstream test_file("./gen_nvida_symbols.py");
    if (test_file.good()) {
        test_file.close();
        return "./gen_nvida_symbols.py";
    }
    
    return "";
}

/**
 * 自动生成符号表文件
 * 
 * @param symbol_file_path 输出符号表文件路径
 * @return true 如果成功生成，false 否则
 */
static bool auto_generate_symbol_table(const std::string& symbol_file_path) {
    // 查找 Python 解释器
    const char* python_cmd = getenv("PYTHON");
    if (!python_cmd || strlen(python_cmd) == 0) {
        // 尝试常见的 Python 命令
        const char* python_commands[] = {"python3", "python", nullptr};
        for (int i = 0; python_commands[i] != nullptr; i++) {
            // 检查命令是否存在（通过 which）
            char test_cmd[256];
            snprintf(test_cmd, sizeof(test_cmd), "which %s > /dev/null 2>&1", python_commands[i]);
            if (system(test_cmd) == 0) {
                python_cmd = python_commands[i];
                break;
            }
        }
    }
    
    if (!python_cmd || strlen(python_cmd) == 0) {
        DEBUG_LOG("Python not found, cannot auto-generate symbol table\n");
        return false;
    }
    
    // 查找脚本路径
    std::string script_path = find_gen_symbols_script();
    if (script_path.empty()) {
        DEBUG_LOG("gen_nvida_symbols.py not found, cannot auto-generate symbol table\n");
        return false;
    }
    
    // 查找 NCCL 库路径
    std::string nccl_lib_path = find_nccl_library_path();
    if (nccl_lib_path.empty()) {
        DEBUG_LOG("NCCL library not found, cannot auto-generate symbol table\n");
        return false;
    }
    
    // 确保输出文件使用 JSON 格式（.json 扩展名）
    std::string output_path = symbol_file_path;
    if (output_path.size() < 5 || output_path.substr(output_path.size() - 5) != ".json") {
        // 如果扩展名不是 .json，添加 .json
        if (output_path.size() > 3 && output_path.substr(output_path.size() - 3) == ".pb") {
            output_path = output_path.substr(0, output_path.size() - 3) + ".json";
        } else {
            output_path = output_path + ".json";
        }
    }
    
    // 构建命令
    // 创建符号表文件的目录（如果不存在）
    size_t last_slash = output_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string dir = output_path.substr(0, last_slash);
        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", dir.c_str());
        system(mkdir_cmd);  // 忽略返回值
    }
    
    // 执行生成脚本
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), 
             "%s %s %s 2>&1",
             python_cmd, script_path.c_str(), output_path.c_str());
    
    // 设置 NCCL_LIB 环境变量（如果未设置）
    char env_cmd[2560];
    const char* nccl_lib_env = getenv("NCCL_LIB");
    if (!nccl_lib_env || strlen(nccl_lib_env) == 0) {
        snprintf(env_cmd, sizeof(env_cmd), 
                 "NCCL_LIB=%s %s", nccl_lib_path.c_str(), cmd);
    } else {
        snprintf(env_cmd, sizeof(env_cmd), "%s", cmd);
    }
    
    // 记录开始时间
    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_REALTIME, &start_ts);
    
    DEBUG_LOG("Auto-generating symbol table: %s\n", env_cmd);
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Starting symbol table generation...\n", get_rank_from_env());
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Command: %s\n", get_rank_from_env(), env_cmd);
    fflush(stderr);
    
    int ret = system(env_cmd);
    
    // 记录结束时间
    clock_gettime(CLOCK_REALTIME, &end_ts);
    double elapsed = (end_ts.tv_sec - start_ts.tv_sec) + (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
    
    if (ret != 0) {
        DEBUG_LOG("Failed to generate symbol table (exit code: %d)\n", ret);
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Symbol table generation failed (exit code: %d, elapsed: %.2f seconds)\n", 
                get_rank_from_env(), ret, elapsed);
        fflush(stderr);
        return false;
    }
    
    // 检查生成的文件是否存在
    std::ifstream test_file(output_path);
    if (!test_file.good()) {
        DEBUG_LOG("Symbol table file was not created: %s\n", output_path.c_str());
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Symbol table file was not created: %s\n", 
                get_rank_from_env(), output_path.c_str());
        fflush(stderr);
        return false;
    }
    test_file.close();
    
    DEBUG_LOG("Successfully auto-generated symbol table: %s\n", output_path.c_str());
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Successfully auto-generated symbol table: %s (elapsed: %.2f seconds)\n", 
            get_rank_from_env(), output_path.c_str(), elapsed);
    fflush(stderr);
    return true;
}

/**
 * 初始化符号表（延迟加载，支持自动生成）
 */
static void ensure_symbols_loaded() {
    if (g_symbols_loaded || g_symbols_load_attempted) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    if (g_symbols_loaded || g_symbols_load_attempted) {
        return;
    }
    
    g_symbols_load_attempted = true;
    
    // 从环境变量获取符号文件路径
    const char* symbol_file = getenv("NCCL_SYMBOL_FILE");
    std::string symbol_file_path;
    
    if (!symbol_file || strlen(symbol_file) == 0) {
        // 尝试默认路径：与 NCCL 库同目录或临时目录
        std::string nccl_lib_path = find_nccl_library_path();
        if (!nccl_lib_path.empty()) {
            size_t last_slash = nccl_lib_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                symbol_file_path = nccl_lib_path.substr(0, last_slash + 1) + "nccl_symbols.json";
            }
        }
        
        // 如果还是空，使用临时目录
        if (symbol_file_path.empty()) {
            const char* tmp_dir = getenv("TMPDIR");
            if (!tmp_dir) tmp_dir = "/tmp";
            char tmp_path[512];
            snprintf(tmp_path, sizeof(tmp_path), "%s/nccl_symbols_%d.json", tmp_dir, getpid());
            symbol_file_path = tmp_path;
        }
    } else {
        symbol_file_path = symbol_file;
        // 如果用户指定的文件没有扩展名，添加 .json
        if (symbol_file_path.size() < 5 || 
            (symbol_file_path.substr(symbol_file_path.size() - 5) != ".json" &&
             symbol_file_path.substr(symbol_file_path.size() - 3) != ".pb")) {
            symbol_file_path = symbol_file_path + ".json";
        }
    }
    
    // 检查符号表文件是否存在
    std::ifstream test_file(symbol_file_path);
    bool file_exists = test_file.good();
    test_file.close();
    
    // 获取当前 rank
    int current_rank = get_rank_from_env();
    bool should_generate = (current_rank % 8 == 0);
    
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] ensure_symbols_loaded: file_exists=%d, should_generate=%d, path=%s\n", 
            current_rank, file_exists ? 1 : 0, should_generate ? 1 : 0, symbol_file_path.c_str());
    fflush(stderr);
    
    // 如果文件不存在，尝试自动生成（仅由 rank%8==0 的进程生成）
    if (!file_exists) {
        const char* auto_gen = getenv("NCCL_AUTO_GEN_SYMBOLS");
        // 如果环境变量未设置或设置为 "1"，则自动生成
        if (!auto_gen || strcmp(auto_gen, "0") != 0) {
            if (should_generate) {
                // rank%8==0 的进程负责生成
                DEBUG_LOG("Rank %d (rank%%8==0) will generate symbol table\n", current_rank);
                fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Will generate symbol table (rank%%8==0)\n", current_rank);
                fflush(stderr);
                
                struct timespec load_start, load_end;
                clock_gettime(CLOCK_REALTIME, &load_start);
                
                if (auto_generate_symbol_table(symbol_file_path)) {
                    file_exists = true;
                    clock_gettime(CLOCK_REALTIME, &load_end);
                    double load_elapsed = (load_end.tv_sec - load_start.tv_sec) + (load_end.tv_nsec - load_start.tv_nsec) / 1e9;
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Symbol table generation completed (total elapsed: %.2f seconds)\n", 
                            current_rank, load_elapsed);
                    fflush(stderr);
                } else {
                    DEBUG_LOG("Auto-generation failed, will use dladdr fallback\n");
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Auto-generation failed, will use dladdr fallback\n", current_rank);
                    fflush(stderr);
                }
            } else {
                // 其他进程等待文件生成（最多等待30秒）
                DEBUG_LOG("Rank %d waiting for symbol table to be generated by rank%%8==0 process...\n", current_rank);
                fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Waiting for symbol table to be generated by rank%%8==0 process...\n", current_rank);
                fflush(stderr);
                
                const int max_wait_seconds = 30;
                const int poll_interval_ms = 100;  // 每100ms检查一次
                int waited_ms = 0;
                
                while (waited_ms < max_wait_seconds * 1000) {
                    std::ifstream check_file(symbol_file_path);
                    if (check_file.good()) {
                        check_file.close();
                        file_exists = true;
                        DEBUG_LOG("Rank %d found symbol table after waiting %d ms\n", current_rank, waited_ms);
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Found symbol table after waiting %d ms\n", 
                                current_rank, waited_ms);
                        fflush(stderr);
                        break;
                    }
                    check_file.close();
                    usleep(poll_interval_ms * 1000);
                    waited_ms += poll_interval_ms;
                    
                    // 每5秒打印一次等待状态
                    if (waited_ms % 5000 == 0) {
                        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Still waiting for symbol table... (%d ms elapsed)\n", 
                                current_rank, waited_ms);
                        fflush(stderr);
                    }
                }
                
                if (!file_exists) {
                    DEBUG_LOG("Rank %d timeout waiting for symbol table, will use dladdr fallback\n", current_rank);
                    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Timeout waiting for symbol table, will use dladdr fallback\n", current_rank);
                    fflush(stderr);
                }
            }
        }
    }
    
    // 尝试加载符号表
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Attempting to load symbol table from %s...\n", 
            current_rank, symbol_file_path.c_str());
    fflush(stderr);
    
    struct timespec parse_start, parse_end;
    clock_gettime(CLOCK_REALTIME, &parse_start);
    
    // 注意：load_symbol_table_protobuf 内部不再获取锁，因为调用者已经持有锁
    if (file_exists && load_symbol_table_protobuf(symbol_file_path)) {
        clock_gettime(CLOCK_REALTIME, &parse_end);
        double parse_elapsed = (parse_end.tv_sec - parse_start.tv_sec) + (parse_end.tv_nsec - parse_start.tv_nsec) / 1e9;
        
        g_symbols_loaded = true;
        DEBUG_LOG("Loaded %zu NCCL kernel symbols from %s\n", 
                  g_offset_to_symbol.size(), symbol_file_path.c_str());
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Loaded %zu NCCL kernel symbols from %s (parse time: %.2f seconds)\n", 
                current_rank, g_offset_to_symbol.size(), symbol_file_path.c_str(), parse_elapsed);
        fflush(stderr);
    } else {
        if (file_exists) {
            DEBUG_LOG("Failed to load symbol table from %s (protobuf support may be missing)\n", 
                      symbol_file_path.c_str());
            fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] Failed to load symbol table from %s\n", 
                    current_rank, symbol_file_path.c_str());
            fflush(stderr);
        }
        // 如果加载失败，is_nccl_kernel_func() 会回退到 dladdr() 方法
    }
    
    fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] ensure_symbols_loaded completed\n", current_rank);
    fflush(stderr);
}

/**
 * 获取 NCCL kernel 的详细信息
 * 
 * @param func kernel 函数指针
 * @param symbol 输出参数，如果找到则填充符号信息
 * @return 1 如果找到且是 NCCL kernel，0 否则
 */
static int get_nccl_kernel_info(const void* func, NcclKernelSymbol* symbol) {
    if (!func || !symbol) {
        return 0;
    }
    
    // 确保符号表已加载
    ensure_symbols_loaded();
    
    // 如果符号表为空，无法获取详细信息
    if (g_offset_to_symbol.empty()) {
        return 0;
    }
    
    // 计算偏移量
    int64_t offset = get_function_offset(func);
    if (offset < 0) {
        return 0;
    }
    
    // 在符号表中查找
    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    auto it = g_offset_to_symbol.find(offset);
    if (it != g_offset_to_symbol.end() && it->second.func_type == "NCCL") {
        *symbol = it->second;
        return 1;
    }
    
    return 0;
}

/**
 * 检查kernel函数是否为NCCL kernel
 * 
 * 实现原理：
 * 1. 优先使用符号表（如果已加载）：通过偏移量查找，可以识别具体的 NCCL 操作
 * 2. 回退到 dladdr() 方法：检查函数地址所在的共享库
 * 
 * 符号表方法（优先）：
 * - 如果已加载符号表，通过计算函数偏移量在符号表中查找
 * - 如果找到且 func_type == "NCCL"，则确认是 NCCL kernel
 * - 优点：可以识别具体的操作类型（AllReduce, AllGather 等）
 * 
 * dladdr() 方法（回退）：
 * - 通过 dladdr() 检查函数所在库的基址和文件名
 * - 如果库基址匹配 NCCL 库或文件名包含 "nccl"，则认为是 NCCL kernel
 * - 缺点：只能判断是否是 NCCL kernel，无法识别具体操作
 */
static int is_nccl_kernel_func(const void* func) {
    if (!func) return 0;
    
    // 确保符号表已加载
    ensure_symbols_loaded();
    
    // 方法1: 优先使用符号表（如果已加载）
    if (!g_offset_to_symbol.empty()) {
        int64_t offset = get_function_offset(func);
        if (offset >= 0) {
            std::lock_guard<std::mutex> lock(g_symbol_mutex);
            auto it = g_offset_to_symbol.find(offset);
            if (it != g_offset_to_symbol.end() && it->second.func_type == "NCCL") {
                return 1;  // 在符号表中找到，确认是 NCCL kernel
            }
        }
    }
    
    // 方法2: 回退到原有的 dladdr() 方法
    Dl_info info;
    if (dladdr(func, &info) == 0) {
        return 0;
    }
    
    // 检查库文件名是否包含 "nccl"（不区分大小写）
    if (info.dli_fname) {
        const char* fname = info.dli_fname;
        char lower_fname[256];
        size_t len = strlen(fname);
        if (len < sizeof(lower_fname)) {
            for (size_t i = 0; i < len; i++) {
                lower_fname[i] = tolower(fname[i]);
            }
            lower_fname[len] = '\0';
            
            if (strstr(lower_fname, "nccl") != nullptr) {
                return 1;  // 找到 "nccl"，确认是 NCCL kernel
            }
        }
    }
    
    // 检查库基址是否在已加载的 NCCL 库中
    void* nccl_handle = get_nccl_handle();
    if (nccl_handle && info.dli_fbase == nccl_handle) {
        return 1;  // 库基址匹配，确认是 NCCL kernel
    }
    
    return 0;
}

// ============================================================================
// NCCL解析功能实现
// ============================================================================

int nccl_get_comm_id(ncclComm_t comm, ncclUniqueId* comm_id) {
    if (!comm || !comm_id) return -1;
    
    std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
    auto it = g_comm_to_comm_id.find(comm);
    if (it != g_comm_to_comm_id.end()) {
        *comm_id = it->second;
        return 0;
    }
    
    memset(comm_id, 0, sizeof(ncclUniqueId));
    return -1;
}

ncclComm_t nccl_get_comm_by_id(const ncclUniqueId* comm_id) {
    if (!comm_id) return nullptr;
    
    std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
    auto it = g_comm_id_to_comm.find(*comm_id);
    if (it != g_comm_id_to_comm.end()) {
        return it->second;
    }
    return nullptr;
}

uint64_t nccl_get_group_hash(ncclComm_t comm) {
    if (!comm) return 0;
    
    ncclUniqueId comm_id;
    if (nccl_get_comm_id(comm, &comm_id) != 0) {
        return 0;
    }
    
    // 检查commId是否为空
    bool is_empty = true;
    const char* bytes = reinterpret_cast<const char*>(&comm_id);
    for (int i = 0; i < static_cast<int>(sizeof(ncclUniqueId)); i++) {
        if (bytes[i] != 0) {
            is_empty = false;
            break;
        }
    }
    
    if (is_empty) return 0;
    
    return hash_unique_id(comm_id);
}

void* nccl_extract_dev_comm(ncclComm_t comm) {
    return extract_dev_comm(comm);
}

int nccl_get_op_info(void* dev_comm, nccl_op_info_t* info) {
    if (!dev_comm || !info) return -1;
    
    std::lock_guard<std::mutex> lock(g_nccl_info_mutex);
    auto it = g_nccl_info_map.find(dev_comm);
    if (it != g_nccl_info_map.end() && !it->second.empty()) {
        const NcclInfo& nccl_info = it->second.front();
        info->count = nccl_info.count;
        info->datatype = nccl_info.datatype;
        info->comm = nccl_info.comm;
        strncpy(info->operation_type, nccl_info.operation_type.c_str(), 
                sizeof(info->operation_type) - 1);
        info->operation_type[sizeof(info->operation_type) - 1] = '\0';
        return 0;
    }
    return -1;
}

// ============================================================================
// NCCL函数拦截实现
// ============================================================================

static void store_nccl_info(ncclComm_t comm, size_t count, ncclDataType_t datatype, const char* op_type) {
    void* dev_comm = extract_dev_comm(comm);
    if (!dev_comm) return;
    
    NcclInfo info;
    info.count = count;
    info.datatype = datatype;
    info.comm = comm;
    info.operation_type = op_type;
    
    std::lock_guard<std::mutex> lock(g_nccl_info_mutex);
    g_nccl_info_map[dev_comm].push(info);
}

static void invoke_nccl_callback(const char* op_name, size_t count, ncclDataType_t datatype,
                                   cudaStream_t stream, ncclComm_t comm) {
    if (!g_intercept_enabled) {
        // 调试：如果拦截未启用，输出信息（仅第一次）
        static int warned = 0;
        if (!warned) {
            DEBUG_LOG("Warning: Intercept disabled, skipping %s\n", op_name);
            warned = 1;
        }
        return;
    }
    
    // 安全地获取 group_hash，如果 comm 还未映射则返回 0
    uint64_t group_hash = 0;
    try {
        group_hash = nccl_get_group_hash(comm);
    } catch (...) {
        // 忽略异常，使用默认值 0
        group_hash = 0;
    }
    
    char timestamp[64];
    get_timestamp_string(timestamp, sizeof(timestamp));
    
    // 写入文件日志
    if (g_log_enabled) {
        write_nccl_log(op_name, count, datatype, stream, comm, group_hash, timestamp);
    } else {
        // 调试：如果日志未启用但拦截已启用，输出警告（仅第一次）
        static int warned_log = 0;
        if (!warned_log) {
            DEBUG_LOG("Warning: Logging disabled but intercept enabled for %s\n", op_name);
            warned_log = 1;
        }
    }
    
    // 调用用户回调（如果设置）
    if (g_nccl_callback) {
        g_nccl_callback(op_name, count, datatype, stream, comm, group_hash, timestamp);
    }
}

extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    ncclGetUniqueId_t func = safe_dlsym("ncclGetUniqueId", &real_ncclGetUniqueId);
    if (!func) {
        return ncclSystemError;
    }
    
    ncclResult_t result = func(uniqueId);
    if (result == ncclSuccess && uniqueId) {
        std::lock_guard<std::mutex> lock(g_unique_id_stack_mutex);
        g_unique_id_stack.push(*uniqueId);
    }
    return result;
}

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    ncclCommInitRank_t func = safe_dlsym("ncclCommInitRank", &real_ncclCommInitRank);
    if (!func) {
        return ncclSystemError;
    }
    
    ncclResult_t result = func(comm, nranks, commId, rank);
    
    if (result == ncclSuccess && comm && *comm) {
        std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
        g_comm_id_to_comm[commId] = *comm;
        g_comm_to_comm_id[*comm] = commId;
        
        {
            std::lock_guard<std::mutex> stack_lock(g_unique_id_stack_mutex);
            if (!g_unique_id_stack.empty() && NcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
                g_unique_id_stack.pop();
            }
        }
    }
    
    return result;
}

extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    ncclCommInitRankConfig_t func = safe_dlsym("ncclCommInitRankConfig", &real_ncclCommInitRankConfig);
    if (!func) {
        return ncclSystemError;
    }
    
    ncclResult_t result = func(comm, nranks, commId, rank, config);
    
    if (result == ncclSuccess && comm && *comm) {
        std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
        g_comm_id_to_comm[commId] = *comm;
        g_comm_to_comm_id[*comm] = commId;
        
        {
            std::lock_guard<std::mutex> stack_lock(g_unique_id_stack_mutex);
            if (!g_unique_id_stack.empty() && NcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
                g_unique_id_stack.pop();
            }
        }
    }
    
    return result;
}

extern "C" ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
    ncclCommInitAll_t func = safe_dlsym("ncclCommInitAll", &real_ncclCommInitAll);
    if (!func) {
        return ncclSystemError;
    }
    
    ncclResult_t result = func(comm, ndev, devlist);
    
    if (result == ncclSuccess && comm) {
        ncclUniqueId commId;
        bool has_comm_id = false;
        {
            std::lock_guard<std::mutex> stack_lock(g_unique_id_stack_mutex);
            if (!g_unique_id_stack.empty()) {
                commId = g_unique_id_stack.top();
                g_unique_id_stack.pop();
                has_comm_id = true;
            }
        }
        
        if (has_comm_id) {
            std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
            for (int i = 0; i < ndev; i++) {
                if (comm[i] != nullptr) {
                    g_comm_id_to_comm[commId] = comm[i];
                    g_comm_to_comm_id[comm[i]] = commId;
                }
            }
        }
    }
    
    return result;
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, 
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    ncclAllReduce_t func = safe_dlsym("ncclAllReduce", &real_ncclAllReduce);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, recvbuff, count, datatype, op, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "AllReduce");
        invoke_nccl_callback("AllReduce", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count, 
                                      ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    ncclAllGather_t func = safe_dlsym("ncclAllGather", &real_ncclAllGather);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, recvbuff, count, datatype, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "AllGather");
        invoke_nccl_callback("AllGather", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count, 
                                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    ncclReduceScatter_t func = safe_dlsym("ncclReduceScatter", &real_ncclReduceScatter);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, recvbuff, count, datatype, op, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "ReduceScatter");
        invoke_nccl_callback("ReduceScatter", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, 
                                    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    ncclReduce_t func = safe_dlsym("ncclReduce", &real_ncclReduce);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "Reduce");
        invoke_nccl_callback("Reduce", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, 
                                       ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    ncclBroadcast_t func = safe_dlsym("ncclBroadcast", &real_ncclBroadcast);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, recvbuff, count, datatype, root, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "Broadcast");
        invoke_nccl_callback("Broadcast", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, 
                                  int peer, ncclComm_t comm, cudaStream_t stream) {
    ncclSend_t func = safe_dlsym("ncclSend", &real_ncclSend);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, count, datatype, peer, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "Send");
        invoke_nccl_callback("Send", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, 
                                 int peer, ncclComm_t comm, cudaStream_t stream) {
    ncclRecv_t func = safe_dlsym("ncclRecv", &real_ncclRecv);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(recvbuff, count, datatype, peer, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, count, datatype, "Recv");
        invoke_nccl_callback("Recv", count, datatype, stream, comm);
    }
    
    return result;
}

extern "C" ncclResult_t ncclSendRecv(const void* sendbuff, size_t sendcount, ncclDataType_t sendtype, 
                                      int peer_send, void* recvbuff, size_t recvcount, ncclDataType_t recvtype, 
                                      int peer_recv, ncclComm_t comm, cudaStream_t stream) {
    ncclSendRecv_t func = safe_dlsym("ncclSendRecv", &real_ncclSendRecv);
    if (!func) {
        return ncclSystemError;
    }
    
    // 先调用真实函数，确保 NCCL 操作正常完成
    ncclResult_t result = func(sendbuff, sendcount, sendtype, peer_send, recvbuff, recvcount, recvtype, peer_recv, comm, stream);
    
    // 只有在成功时才记录信息和调用回调
    if (g_intercept_enabled && result == ncclSuccess) {
        store_nccl_info(comm, sendcount, sendtype, "SendRecv");
        invoke_nccl_callback("SendRecv", sendcount, sendtype, stream, comm);
    }
    
    return result;
}

// ============================================================================
// CUDA Kernel拦截实现
// ============================================================================

extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                                        void** args, size_t sharedMem, cudaStream_t stream) {
    cudaLaunchKernel_t func_ptr = safe_dlsym("cudaLaunchKernel", &real_cudaLaunchKernel);
    if (!func_ptr) {
        return cudaErrorUnknown;
    }
    
    // 调用真实函数
    cudaError_t result = func_ptr(func, gridDim, blockDim, args, sharedMem, stream);
    
    // 调用回调（如果启用）
    if (g_intercept_enabled && result == cudaSuccess) {
        char timestamp[64];
        get_timestamp_string(timestamp, sizeof(timestamp));
        
        // 检查是否为NCCL kernel
        // 通过 dladdr() 检查 kernel 函数地址所在的共享库来判断
        int is_nccl_kernel = is_nccl_kernel_func(func);
        
        // 写入文件日志
        if (g_log_enabled) {
            write_kernel_log(func, gridDim, blockDim, stream, is_nccl_kernel, timestamp);
        }
        
        // 调用用户回调（如果设置）
        if (g_kernel_callback) {
            g_kernel_callback(func, gridDim, blockDim, stream, is_nccl_kernel, timestamp);
        }
    }
    
    return result;
}

// ============================================================================
// 初始化和清理
// ============================================================================

int nccl_cuda_intercept_init(nccl_op_callback_t nccl_cb, cuda_kernel_callback_t kernel_cb) {
    g_nccl_callback = nccl_cb;
    g_kernel_callback = kernel_cb;
    g_intercept_enabled = 1;
    return 0;
}

void nccl_cuda_intercept_cleanup(void) {
    g_nccl_callback = nullptr;
    g_kernel_callback = nullptr;
    g_intercept_enabled = 0;
    
    // 关闭日志文件
    close_log_file();
    
    std::lock_guard<std::mutex> lock1(g_comm_mapping_mutex);
    std::lock_guard<std::mutex> lock2(g_nccl_info_mutex);
    g_comm_id_to_comm.clear();
    g_comm_to_comm_id.clear();
    g_nccl_info_map.clear();
    
    // 关闭 NCCL 库句柄
    std::lock_guard<std::mutex> lock3(g_nccl_handle_mutex);
    if (g_nccl_handle && g_nccl_handle != RTLD_NEXT) {
        dlclose(g_nccl_handle);
        g_nccl_handle = nullptr;
    }
}

void nccl_cuda_intercept_set_enabled(int enable) {
    g_intercept_enabled = enable ? 1 : 0;
}

void nccl_set_dev_comm_extractor(void* (*extractor)(ncclComm_t)) {
    g_dev_comm_extractor = extractor;
}

// ============================================================================
// 库加载时自动初始化
// ============================================================================

/**
 * 库加载时自动调用的初始化函数
 * 使用 __attribute__((constructor)) 确保在库被加载时自动执行
 */
__attribute__((constructor))
static void auto_init_on_load() {
    // 使用try-catch保护，避免段错误
    try {
        // 检查是否启用调试输出（必须在最前面，避免使用未初始化的变量）
        const char* debug_str = getenv("NCCL_INTERCEPT_DEBUG");
        if (debug_str && strcmp(debug_str, "debug") == 0) {
            g_debug_enabled = 1;
        }
        
        // 输出初始化信息（用于调试）
        DEBUG_LOG("Library loaded, initializing...\n");
        
        // 初始化日志系统（如果环境变量启用）
        init_log_system();
        
        // 如果启用了日志，自动设置拦截为启用状态
        if (g_log_enabled) {
            g_intercept_enabled = 1;
            const char* log_path_str = getenv("NCCL_MEGATRACE_LOG_PATH");
            const char* log_path = (log_path_str && strlen(log_path_str) > 0) ? log_path_str : "./mega_log";
            DEBUG_LOG("Logging enabled for rank %d, log path: %s\n", 
                    g_rank, log_path);
        } else {
            // 即使日志未启用，也输出信息以便调试
            const char* enable_str = getenv("NCCL_MEGATRACE_ENABLE");
            DEBUG_LOG("Logging disabled (NCCL_MEGATRACE_ENABLE=%s)\n", 
                    enable_str ? enable_str : "not set");
        }
        
        fprintf(stderr, "[NCCL_INTERCEPT] [Rank %d] auto_init_on_load completed\n", g_rank);
        fflush(stderr);
    } catch (...) {
        // 如果构造函数中出现异常，至少确保拦截功能可以工作
        // 不输出任何信息，避免进一步的问题
        g_log_enabled = 0;
        g_intercept_enabled = 1;  // 仍然启用拦截，只是不记录日志
        fprintf(stderr, "[NCCL_INTERCEPT] Exception in auto_init_on_load\n");
        fflush(stderr);
    }
}

/**
 * 库卸载时自动调用的清理函数
 * 使用 __attribute__((destructor)) 确保在库被卸载时自动执行
 */
__attribute__((destructor))
static void auto_cleanup_on_unload() {
    // 关闭日志文件
    close_log_file();
}


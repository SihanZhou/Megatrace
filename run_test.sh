export NCCL_MEGATRACE_ENABLE=1
export NCCL_DEBUG=version
export NCCL_MEGATRACE_LOG_PATH=./mega_log
export NCCL_INTERCEPT_DEBUG=debug
#export LD_PRELOAD=/workspace/jjj/nccl-intercept-1300/nccl_intercept_0102.so
export LD_PRELOAD=./libnccl_cuda_intercept.so
torchrun \
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr 127.0.0.1 \
--master_port 22234 test.py

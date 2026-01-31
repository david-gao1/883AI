#!/bin/bash
# 激活conda环境并设置环境变量
source /Users/lianggao/miniforge3/etc/profile.d/conda.sh
conda activate 883ai

# 设置OpenMP环境变量（解决macOS上的OpenMP问题）
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# 运行主程序
cd "$(dirname "$0")"
python main.py "$@"

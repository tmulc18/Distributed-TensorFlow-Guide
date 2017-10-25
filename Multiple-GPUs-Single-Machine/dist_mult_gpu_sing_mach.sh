#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
python dist_mult_gpu_sing_mach.py --job_name "ps" --task_index 0 &
export CUDA_VISIBLE_DEVICES=0
python dist_mult_gpu_sing_mach.py --job_name "worker" --task_index 0 &
export CUDA_VISIBLE_DEVICES=1
python dist_mult_gpu_sing_mach.py --job_name "worker" --task_index 1 &

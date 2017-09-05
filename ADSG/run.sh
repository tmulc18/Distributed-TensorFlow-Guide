#!/bin/bash
python dist_cpu_sing_mach_sync.py --job_name "ps" --task_index 0 &
python dist_cpu_sing_mach_sync.py --job_name "worker" --task_index 0 &
python dist_cpu_sing_mach_sync.py --job_name "worker" --task_index 1 &
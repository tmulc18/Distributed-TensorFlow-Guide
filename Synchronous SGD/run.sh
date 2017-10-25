#!/bin/bash
python ssgd.py --job_name "ps" --task_index 0 &
python ssgd.py --job_name "worker" --task_index 0 &
python ssgd.py --job_name "worker" --task_index 1 &
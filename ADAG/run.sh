#!/bin/bash
python ADAG.py --job_name "ps" --task_index 0 &
python ADAG.py --job_name "worker" --task_index 0 &
python ADAG.py --job_name "worker" --task_index 1 &
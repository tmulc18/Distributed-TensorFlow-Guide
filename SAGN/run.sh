#!/bin/bash
python SAGN.py --job_name "ps" --task_index 0 &
python SAGN.py --job_name "worker" --task_index 0 &
python SAGN.py --job_name "worker" --task_index 1 &

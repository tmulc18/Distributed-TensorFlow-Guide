#!/bin/bash
python DOWNPOUR.py --job_name "ps" --task_index 0 &
python DOWNPOUR.py --job_name "worker" --task_index 0 &
python DOWNPOUR.py --job_name "worker" --task_index 1 &
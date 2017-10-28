#!/bin/bash
python AGN.py --job_name "ps" --task_index 0 &
python AGN.py --job_name "worker" --task_index 0 &
python AGN.py --job_name "worker" --task_index 1 &
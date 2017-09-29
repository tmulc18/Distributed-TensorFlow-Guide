#!/bin/bash
python Hogwild.py --job_name "ps" --task_index 0 &
python Hogwild.py --job_name "worker" --task_index 0 &
python Hogwild.py --job_name "worker" --task_index 1 &

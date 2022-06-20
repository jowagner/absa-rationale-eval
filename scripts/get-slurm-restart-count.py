#!/usr/bin/env python

import os
import sys

worker_id = []
try:
    worker_id.append(os.environ['SLURM_JOB_ID'])
except KeyError:
    pass
try:
    worker_id.append(os.environ['SLURM_JOB_NODELIST'])
except KeyError:
    worker_id.append(os.uname()[1])
try:
    worker_id.append(os.environ['SLURM_TASK_PID'])
except KeyError:
    worker_id.append('%d' %os.getpid())

try:
    count = os.environ['SLURM_RESTART_COUNT']
except KeyError:
    count = 0

format_str = '%%0%dd\n' %int(sys.argv[1])
worker_id.append(format_str %int(count))

sys.stdout.write('-'.join(worker_id))


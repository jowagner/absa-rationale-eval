#!/usr/bin/env python

import os
import sys

try:
    count = os.environ['SLURM_RESTART_COUNT']
except KeyError:
    count = 0

format_str = '%%0%dd\n' %int(sys.argv[1])

sys.stdout.write(format_str %int(count))


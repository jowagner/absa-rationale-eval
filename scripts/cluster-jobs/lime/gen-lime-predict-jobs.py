#!/usr/bin/env python

import sys

outsuffix = 'p%d' %int(sys.argv[1])

template = open('template-lime-predict.job', 'r').read()

gpus = [
    ('rtx6000',   256, 88.5, 18.0, '#SBATCH -t 19:40:00'),
    ('rtx3090',   256, 77.5, 20.0, '#SBATCH -t 21:50:00'),
    #('titanv',     64, 77.5, 40.0, '#SBATCH --reservation=themea'),  # issue with cuda env
    ('rtx2080ti', 128, 77.5, 20.0, '#SBATCH -t 21:50:00'),
]
gpu_index = 0

for shortname, longname in [
    ('f', 'Full'),
]:
    for part in (1,2,3,4):
      for run in (1,2,3):
        gpu, batchsize, speed, hours, more_sbatch = gpus[gpu_index]
        f = open('run-lime-predict-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()
        gpu_index += 1
        if gpu_index == len(gpus):
            gpu_index = 0

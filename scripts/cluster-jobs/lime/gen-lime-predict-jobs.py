#!/usr/bin/env python

import sys

outsuffix = 'p%d' %int(sys.argv[1])

template = open('template-lime-predict.job', 'r').read()

gpus = [
    ('rtx6000',   256, ''),
    ('rtx3090',   256, ''),
    #('titanv',     64, '#SBATCH --reservation=themea'),  # issue with cuda env
    ('rtx2080ti', 128, ''),
]
gpu_index = 0

for shortname, longname in [
    ('f', 'Full'),
]:
    for part in (1,2,3,4):
      for run in (1,2,3):
        gpu, batchsize, more_sbatch = gpus[gpu_index]
        f = open('run-lime-predict-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()
        gpu_index += 1
        if gpu_index == len(gpus):
            gpu_index = 0

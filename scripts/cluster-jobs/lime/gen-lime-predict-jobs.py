#!/usr/bin/env python

import sys

outsuffix = 'p%s' %(sys.argv[1])

template = open('template-lime-predict.job', 'r').read()

gpus = [
    ('rtx6000',   256, 80.5, 6.0, '#SBATCH -t 07:55:00'),
    ('rtx3090',   256, 77.5, 20.0, '#SBATCH -t 21:50:00'),
    #('titanv',     64, 77.5, 40.0, '#SBATCH --reservation=themea'),  # issue with cuda env
    ('rtx2080ti', 128, 77.5, 20.0, '#SBATCH -t 21:50:00'),
]
gpu_index = 0

shortname, longname = ('f', 'Full')

for gpu, batchsize, speed, hours, more_sbatch in gpus:   # [gpu_index]
    # TODO: add option to add --requeue and --qos=preempt
    for part in (1,2,3,4):
      for run in (1,2,3):
        #if (part, run) not in [(3,1), (4,3), ]:
        #    continue
        f = open('run-lime-predict-%(outsuffix)s-c-%(shortname)s-%(part)d%(run)d-%(gpu)s.job' %locals(), 'w')
        f.write(template %locals())
        f.close()
        gpu_index += 1
        if gpu_index == len(gpus):
            gpu_index = 0

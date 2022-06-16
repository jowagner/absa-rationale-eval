#!/usr/bin/env python

import sys

outsuffix = 'p%d' %int(sys.argv[1])

template = open('template-lime-predict.job', 'r').read()

for shortname, longname in [
    ('f', 'Full'),
]:
    for part, gpu, batchsize, more_sbatch in [
        (1, 'rtx6000',   256, ''),
        (2, 'rtx3090',   256, ''),
        (3, 'titanv',     64, '#SBATCH --reservation=themea'),
        (4, 'rtx2080ti', 128, ''),
    ]:
      for run in (1,2,3):
        f = open('run-lime-predict-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()

#!/usr/bin/env python

import sys

outsuffix = 'p%d' %int(sys.argv[1])

template = open('template-lime-predict.job', 'r').read()

for shortname, longname in [
    ('f', 'Full'),
]:
    for part in (1,2,3,4):
      for run, gpu, batchsize in [
          (1, 'rtx2080ti', 128),
          (2, 'rtx6000',   256),
          (3, 'rtx3090',   256),
      ]:
        f = open('run-lime-predict-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()

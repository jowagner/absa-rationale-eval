#!/usr/bin/env python

import sys

outsuffix = 'p%d' %int(sys.argv[1])

template = open('template-lime-preload.job', 'r').read()

for shortname, longname in [
    ('f', 'Full'),
]:
    for part in (1,2,3,4):
      for run in (1,2,3):
        f = open('run-lime-preload-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()

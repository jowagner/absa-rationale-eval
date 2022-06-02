#!/usr/bin/env python

import sys

template = open('template-saliency.job', 'r').read()

if len(sys.argv) > 1 and sys.argv[1] == 'sevenpoints':
    outsuffix = 'sevenpoints-xfw'
    options   = '--load-model-from best-sea.ckpt --gradient-method seven_points'
elif len(sys.argv) > 1 and sys.argv[1] == 'shortline-test':
    outsuffix = 'shortline-test-xfw'
    options   = '--load-model-from best-sea.ckpt --gradient-method short_line --test-saliencies-only'
else:
    outsuffix = 'morewio-xfw'
    options   = '--load-model-from best-sea.ckpt'

# speed-up when saliency scores are available from a previous run:
# --saliencies-from saliency-tr-stdout.txt --saliencies-from saliency-stdout-wfw.txt'

for shortname, longname in [
    ('f', 'Full'),
    #('s', 'SE'),
    #('o', 'Other'),
    #('a', 'All'),
]:
    for part in (1,2,3,4):
      for run in (1,2,3):
        f = open('run-saliency-c-%(shortname)s-%(part)d%(run)d.job' %locals(), 'w')
        f.write(template %locals())
        f.close()

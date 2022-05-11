#!/usr/bin/env python

template = open('template-saliency.job', 'r').read()

outsuffix = 'morewio-xfw'
options   = '--load-model-from best-sea.ckpt'

# speed-up when saliency scores are available from a previous run:
# --saliencies-from saliency-tr-stdout.txt --saliencies-from saliency-stdout-wfw.txt'

for shortname, longname in [
    ('f', 'Full'),
    ('s', 'SE'),
    ('o', 'Other'),
    #('a', 'All'),
]:
    for part in (1,2,3):
        f = open('run-saliency-c-%(shortname)s-%(part)d1-to-%(part)d3.job' %locals(), 'w')
        f.write(template %locals())
        f.close()

#!/usr/bin/env python

template = open('template-saliency.job', 'r').read()

for shortname, longname in [
    ('f', 'Full'),
    ('s', 'SE'),
    ('o', 'Other'),
    ('a', 'All'),
]:
    for part in (1,2,3):
        f = open('run-saliency-c-%(shortname)s-%(part)d1-to-%(part)d3.job' %locals(), 'w')
        f.write(template %locals())
        f.close()
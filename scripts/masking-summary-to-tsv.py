#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

tr = sys.argv[1]
m_type = sys.argv[2]

assert m_type in ('SE', 'R')

example = """
==> c-f-1-1/stdout-training-with-local-aio.txt <==

Summary:
SeqB                  Q                     Overall               Laptop                Restaurant            Description
Full                  0                     83.834135532          81.930691004          85.686725378          Sun et al. QA-M
SE                    0                     81.730771065          79.084157944          84.297835827          Sun et al. QA-M
Other                 0                     63.822120428          56.064355373          70.640432835          Sun et al. QA-M

==> c-f-1-2/stdout-training-with-local-aio.txt <==

Summary:
SeqB                  Q                     Overall               Laptop                Restaurant            Description
Full     """

run = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    elif line.startswith('==>'):
        run += 1
    elif 'Sun et al. QA-M' in line:
        if line.startswith('Full'):
            te = 'Full'
        elif line.startswith('SE'):
            te = m_type
        elif line.startswith('Other'):
            te = 'Z-CompR' if m_type == 'R' else 'Z-Other'
        else:
            raise ValueError(line)
        for index, domain in enumerate('Laptop Restaurant'.split()):
            score = line.split('\t')[3+index]
            row = []
            row.append(tr)
            row.append(domain)
            row.append(te)
            row.append('%d' %run)
            row.append(score)
            sys.stdout.write('\t'.join(row))
            sys.stdout.write('\n')

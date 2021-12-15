#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

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

trshort2sortkey = {
    'f': 'tr=Full',
    's': 'tr=SE/R',
    'o': 'tr=Y_Other',
    'a': 'tr=Z_Concat',
}

run = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    elif line.startswith('==>'):
        fields = line.replace('/', ' ').split()
        #     [1]     [2]
        # ==> c-f-1-2/stdout-training-with-local-aio.txt <==
        assert len(fields) == 4
        folder = fields[1]
        filename = fields[2]
        fields = folder.split('-')
        assert len(fields) == 4
        assert fields[0] == 'c'
        run = 3*(int(fields[2])-1) + int(fields[3])
        tr = trshort2sortkey[fields[1]]
        if filename == 'stdout.txt':
            m_type = 'tab2-SE'
        elif 'with-union-aio' in filename:
            m_type = 'tab2-U-SE'
        elif filename.startswith('stdout-training-with-L'):
            fields = filename.replace('-', ' ').split()
            aio_name = fields[3].replace('.', ' ').split()[0]
            m_type = 'tab3-' + aio_name
        elif filename == 'stdout-training-with-local-aio.txt':
            m_type = 'tab4-old-R'
        else:
            raise ValueError('unsupported path %s/%s' %(folder, filename))
    elif line.startswith('SeqB'):
        fields = line.split('\t')
        # check column header
        assert fields[3] == 'Laptop'
        assert fields[4] == 'Restaurant'
        assert fields[5].rstrip() == 'Description'
    elif 'Sun et al. QA-M' in line:
        if line.startswith('Full'):
            te = 'Full'
        elif line.startswith('SE'):
            te = 'SE/R'
        elif line.startswith('Other'):
            te = 'Z-CompSE/R'
        else:
            raise ValueError(line)
        for index, domain in enumerate('Laptop Restaurant'.split()):
            score = line.split('\t')[3+index]
            row = []
            row.append(m_type)
            row.append(tr)
            row.append(domain)
            row.append(te)
            row.append('%d' %run)
            row.append(score)
            sys.stdout.write('\t'.join(row))
            sys.stdout.write('\n')

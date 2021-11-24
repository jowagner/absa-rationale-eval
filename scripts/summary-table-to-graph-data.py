#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

example = """
=== Final summary for ('-', 'q0', 'test', None) ==

For 1660 of 1660 test items

From	To	tn	fp	fn	tp	Pr	Re	F	Acc	Avg-Pr	Avg-Re	Avg-F	Avg-Acc	IW-tn	IW-fp	IW-fn	IW-tp	IW-Pr	IW-Re	IW-F	IW-Acc
  0.0	  0.6	8335	0	4003	0	 100.000000000	   0.000000000	   0.000000000	  67.555519533	 100.000000000	   4.518072289	   4.518072289	  56.202197611	400.529692	0.000000	301.161876	0.000000	 100.000000000	   0.000000000	   0.000000000	  57.080590729
  0.7	  0.8	8309	26	4003	0	   0.000000000	   0.000000000	   0.000000000	  67.344788458	  98.433734940	   4.518072289	   4.518072289	  56.164905585	400.196359	0.333333	301.161876	0.000000	   0.000000000	   0.000000000	   0.000000000	  57.033086477
  0.9	  0.9	8303	32	4001	2	   5.882352941	   0.049962528	   0.099083478	  67.312368293	  98.072289157	   4.535283993	   4.548192771	  56.158107232	400.094664	0.435028	301.126161	0.035714	   7.586798114	   0.011858834	   0.023680652	  57.023683395
  1.0	  1.0	8301	34	4000	3	   8.108108108	   0.074943792	   0.148514851	  67.304263252	  97.951807229	   4.595524957	   4.608433735	  56.153618690	400.056928	0.472764	301.107980	0.053896	  10.233563071	   0.017896058	   0.035729633	  57.020896696
  1.1	  1.1	8300	35	3997	6	  14.634146341	   0.149887584	   0.296735905	  67.320473334	  97.891566265	   4.629828839	   4.663989290	  56.158976088	400.036520	0.493172	301.046755	0.115121	  18.925192499	   0.038225487	   0.076296868	  57.026713543
  """

def get_int_x10(s):
    parts = s.split('.')
    whole, fraction = parts
    assert len(fraction) == 1
    if whole == '0':
        whole == ''
    return int(whole+fraction)

data = {}

def read_set(folder, run):
    global data
    f = open('%s/final-test-summary.txt' %folder, 'rt')
    line = f.readline()
    assert line.startswith('=== Final summary for (')
    assert 'test' in line
    line = f.readline()
    assert line.isspace()
    line = f.readline()
    assert line.startswith('For ')
    line = f.readline()
    assert line.isspace()
    line = f.readline()
    assert line.startswith('From')
    header = line.split()
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('#'):
            continue
        if line.isspace() or line.startswith('From'):
            break
        fields = line.split()
        start = get_int_x10(fields[0])
        end   = get_int_x10(fields[1])
        for length in range(start, end+1):
            for c_index in range(2, len(header)):
                c_header = header[c_index]
                value = fields[c_index]
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
                key = (run, length, c_header)
                data[key] = value
    f.close()

for i in [1,2,3]:
    for j in [1,2,3]:
        run = 3*(i-1)+j-1
        read_set('c-f-%d-%d' %(i,j), run)

# data[(run, length, c_header)] with c_header in 
# 'tn fp fn tp Pr Re F Acc Avg-Pr Avg-Re Avg-F Avg-Acc IW-tn IW-fp IW-fn IW-tp IW-Pr IW-Re IW-F IW-Acc'.split()

# write P-R data

for p_header, r_header, description in [
    ('Pr',     'Re',     'micro-per-token'),
    ('Avg-Pr', 'Avg-Re', 'macro-average'),
    ('IW-Pr',  'IW-Re',  'inv-weighted-micro'),
]:
    f = open('PR-%s.tsv' %description, 'wt')
    for (start, end, extra_offset) in [
        (0, 199, 0),
        (200, 400, 9),
        (401,1000, 18),
    ]:
        for length in range(start, end+1):
            for run in range(9):
                row = []
                value = data[(run, length, r_header)]
                row.append('%.9f' %value)
                for _ in range(run+extra_offset):
                    row.append('')
                value = data[(run, length, p_header)]
                row.append('%.9f' %value)
                f.write('\t'.join(row))
                f.write('\n')

    f.close()

# write F-score data (over rationale length)

for f_header, description in [
    ('F',     'micro-per-token'),
    ('Avg-F', 'macro-average'),
    ('IW-F',  'inv-weighted-micro'),
]:
    f = open('FL-%s.tsv' %description, 'wt')
    for length in range(1001):
        row = []
        row.append('%d' %length)
        for run in range(9):
            value = data[(run, length, f_header)]
            row.append('%.9f' %value)
        f.write('\t'.join(row))
        f.write('\n')
    f.close()


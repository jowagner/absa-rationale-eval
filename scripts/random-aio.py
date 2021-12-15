#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

debug = False

if sys.argv[1] == '--seed':
    seed = sys.argv[2]
    del sys.argv[2]
    del sys.argv[1]
    if seed != '0':   # 0 = use system default
        import hashlib
        seed = int(hashlib.sha512(seed).hexdigest(), 16)  # convert string to int conistently across python versions
        if debug: sys.stdout.write('Seed hashed to %d\n' %seed)
        random.seed(seed)

relative_length = float(sys.argv[1])

assert 0.0 <= relative_length <= 1.0

def get_annotation(f):  # TODO: move shared function to a module
    tokens = []
    sea = []
    line_no = 0
    starts_at_line = -1
    while True:
        line = f.readline()
        line_no += 1
        if line.isspace() or not line:
            if tokens:
                yield (tokens, sea, starts_at_line)
                tokens = []
                sea = []
                starts_at_line = -1
            if not line:
                break
        else:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError('Unexpected AIO line %d: %r' %(line_no, line))
            tokens.append(fields[0])
            sea.append(fields[1])
            if starts_at_line < 0:
                starts_at_line = line_no

item_count = 0
for tokens, sea, start_line in get_annotation(sys.stdin):
    item_count += 1
    if debug: sys.stdout.write('\nRead item %d from line %d: %r, %r\n' %(item_count, start_line, tokens, sea))
    n = len(tokens)
    se_length = int(n * relative_length + 0.5)  # rounding to nearest length
    remaining_i_tags = se_length
    remaining_o_tags = n - se_length
    for token in tokens:
        assert remaining_i_tags + remaining_o_tags > 0
        if not remaining_i_tags:
            tag = 'O'
        elif not remaining_o_tags:
            tag = 'I'
        elif random.randrange(remaining_i_tags + remaining_o_tags) < remaining_i_tags:
            tag = 'I'
        else:
            tag = 'O'
        if debug: sys.stdout.write('%d\t%d\t' %(remaining_i_tags, remaining_o_tags))
        sys.stdout.write('%s\t%s\n' %(token, tag))
        if tag == 'I':
            remaining_i_tags -= 1
        else:
            remaining_o_tags -= 1
    sys.stdout.write('\n')
if debug: sys.stdout.write('Done\n')

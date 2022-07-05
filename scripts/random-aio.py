#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

def usage():
    print('Usage: $0 [options] <RELATIVE-LENGTH> < input.aio > output.aio')
    # TODO: print more details how to use this script
    #       input.aio only needs the text column

debug = False
while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--debug':
        debug = True
    elif option in ('--seed', '--random-seed'):
        seed = sys.argv[1]
        del sys.argv[1]
        if seed == '0':
            # 0 = use system default
            if debug: sys.stdout.write('PRNG not seeded\n')
        else:
            import hashlib
            if type(b'') is not str:
                # Python 3
                seed = seed.encode('UTF-8')
            seed = int(hashlib.sha512(seed).hexdigest(), 16)  # convert string to int consistently across Python versions
            if debug: sys.stdout.write('Seed hashed to %d\n' %seed)
            random.seed(seed)
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

relative_length = float(sys.argv[1])

assert 0.0 <= relative_length <= 1.0

def get_annotation(f):  # TODO: move shared function to a module, add options "text_only" and "expected_columns"
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
            #if len(fields) != 2:
            #    raise ValueError('Unexpected AIO line %d: %r' %(line_no, line))
            tokens.append(fields[0])
            #sea.append(fields[1])
            if starts_at_line < 0:
                starts_at_line = line_no

item_count = 0
for tokens, _, start_line in get_annotation(sys.stdin):
    item_count += 1
    if debug: sys.stdout.write('\nRead item %d from line %d: %r\n' %(item_count, start_line, tokens))
    n = len(tokens)
    se_length = int(n * relative_length + 0.5)  # rounding to nearest length
    if debug: sys.stdout.write('%d tokens, %d selected for SE\n' %(n, se_length))
    # assign randomised saliency scores (with index as tie-breaker)
    scores = []
    for index in range(n):
        scores.append((random.random(), index))
    if debug: sys.stdout.write('Saliency map %r\n' %scores)
    # select lowest scoring tokens (interpreting the number as 1-probability)
    # as rationale
    scores.sort()
    selected = list(map(lambda x: x[1], scores[:se_length]))
    if debug: sys.stdout.write('Selected indices %r\n' %selected)
    # write aio format
    count_i = 0
    for index, token in enumerate(tokens):
        if index in selected:
            tag = 'I'
            count_i += 1
        else:
            tag = 'O'
        sys.stdout.write('%s\t%s\n' %(token, tag))
    sys.stdout.write('\n')
    assert count_i == se_length
if debug: sys.stdout.write('Done\n')

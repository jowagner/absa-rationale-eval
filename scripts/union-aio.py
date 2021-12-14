#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

debug = False

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

last_tokens = None
last_count = -1
item_count = 0
for tokens, sea, start_line in get_annotation(sys.stdin):
    item_count += 1
    if debug: sys.stdout.write('\nRead item %d from line %d: %r, %r\n' %(item_count, start_line, tokens, sea))
    if tokens == last_tokens:
        new_sea = []
        for last_tag, tag in zip(last_sea, sea):
            new_sea.append('I' if 'I' in (last_tag, tag) else 'O')
        if debug: sys.stdout.write('Merge:\n%r\n%r\n%r\n' %(last_sea, sea, new_sea))
        last_sea = new_sea
        last_count += 1
    else:
        if debug: sys.stdout.write('Is new\n')
        if last_tokens:
            if debug: sys.stdout.write('Printing %d copies\n' %last_count)
            for _ in range(last_count):
                for token, tag in zip(last_tokens, last_sea):
                    sys.stdout.write('%s\t%s\n' %(token, tag))
                sys.stdout.write('\n')
        else:
            if debug: sys.stdout.write('Nothing to print, last_count = %d\n' %last_count)
        last_tokens = tokens
        last_sea    = sea
        last_count  = 1
if debug: sys.stdout.write('Last item read\n')
if last_tokens:
    if debug: sys.stdout.write('Printing %d copies\n' %last_count)
    for _ in range(last_count):
        for token, tag in zip(last_tokens, last_sea):
            sys.stdout.write('%s\t%s\n' %(token, tag))
        sys.stdout.write('\n')
    # TODO: While our tools can handle .aio files with a trailing empty line,
    #       other people's tools may not as the SEA .aio files do not end
    #       with empty lines. Better to remove the last empty line.
elif debug:
    sys.stdout.write('Nothing to print\n')

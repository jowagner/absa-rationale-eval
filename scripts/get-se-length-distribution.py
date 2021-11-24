#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import random
import sys
import time

def usage():
    print('Usage: $0 [options] <train|test>')
    # TODO: print more details how to use this script

aio_prefix = None
while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--local-aio':
        aio_prefix = 'local-aio/'
    elif option == '--aio-prefix':
        aio_prefix = sys.argv[1]
        del sys.argv[1]
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

dataset_index = ['train', 'test'].index(sys.argv[1])

# TODO: move code shared with train-classifier.py into module

# Dataset Configuration

domains = ['laptop', 'restaurant']

data_prefix = 'data/'
if aio_prefix is None:
    aio_prefix = data_prefix

filenames = {
    'laptop':     ( aio_prefix  + 'train.laptop.aio',
                    aio_prefix  + 'test.laptop.aio'
                  ),

    'restaurant': ( aio_prefix  + 'train.restaurant.aio',
                    aio_prefix  + 'test.restaurant.aio'
                  ),
}

for domain in domains:
    filename = filenames[domain][dataset_index]
    print('Using', filename)

# Get Data Instances AIO Files

def get_annotation(aio_filename):
    f = open(aio_filename, 'r')
    tokens = []
    sea = []
    line_no = 0
    while True:
        line = f.readline()
        line_no += 1
        if line.isspace() or not line:
            if tokens:
                yield (tokens, sea)
                tokens = []
                sea = []
            if not line:
                break
        else:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError('Unexpected AIO line %d: %r' %(line_no, line))
            tokens.append(fields[0])
            sea.append(fields[1])
    f.close()

def get_per_item_stats(aio_filename):
    stats = []
    for tokens, sea in get_annotation(aio_filename):
        sea = ''.join(sea)
        assert len(tokens) == len(sea)
        se_length = sea.count('I')
        sea = sea.replace('O', ' ')
        se_spans = len(sea.split())
        stats.append((
            len(tokens),
            se_length,    
            se_spans,
            sea,
        ))
    return stats

# get data stats

rlen_counts = 11 * [0]
domain2rlen_counts = {}

for domain in domains:
    drlen_counts = 11 * [0]
    aio_filename = filenames[domain][dataset_index]
    stats = get_per_item_stats(aio_filename)
    for item in stats:
        n_tokens, se_length, n_spans, sea = item
        relative_length = se_length/float(n_tokens)
        print('%d\t%d\t%.9f\t%d\t%s' %(
            n_tokens, se_length,
            relative_length,
            n_spans,
            sea.replace(' ', '_')
        ))
        rlen_bin = 10 * se_length // n_tokens
        rlen_counts[rlen_bin] += 1
        drlen_counts[rlen_bin] += 1
    domain2rlen_counts[domain] = drlen_counts

print(rlen_counts)
for domain in domains:
    print(domain, domain2rlen_counts[domain])

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

rlen_counts = []
span_counts = []
domain2rlen_counts = {}
domain2span_counts = {}

def incr_bin(bins, index):
    while len(bins) <= index:
        bins.append(0)
    bins[index] += 1

dataset_size = 0
for domain in domains:
    drlen_counts = []
    dspan_counts = []
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
        incr_bin(rlen_counts, rlen_bin)
        incr_bin(drlen_counts, rlen_bin)
        incr_bin(span_counts, n_spans)
        incr_bin(dspan_counts, n_spans)
        dataset_size += 1
    domain2rlen_counts[domain] = drlen_counts
    domain2span_counts[domain] = dspan_counts

print('Dataset size', dataset_size)

print(rlen_counts)
print(span_counts)
for domain in domains:
    print('rlen', domain, domain2rlen_counts[domain])
    print('span', domain, domain2span_counts[domain])

print('Overall span counts:')
for i in range(100):
    total = 0
    for domain in domains:
        try:
            total += domain2span_counts[domain][i]
        except IndexError:
            pass
    print(i, total) # , '%.1f%%' %(100.0 * total / float(dataset_size)))

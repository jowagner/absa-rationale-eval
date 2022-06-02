#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import os
import random
import sys

def usage():
    print('Usage: $0 [options]')
    # TODO: print more details how to use this script

opt_seed = b'101'
opt_verbose = False
opt_data_dir = 'hyper-parameter-search'
opt_epochs = None
while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h', '-n'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--data-dir':
        opt_data_dir = sys.argv[1]
        del sys.argv[1]
    elif option == '--epochs':
        opt_epochs = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--seed':
        opt_seed = sys.argv[1].encode('utf-8')
        del sys.argv[1]
        if not opt_seed:
            # use system randomness as non-deterministic seed
            opt_seed = b'%064x' %random.getrandbits(256)
    elif option == '--verbose':
        opt_verbose = True
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

model_name = 'training-with-sea-aio'
log_name = 'stdout-%s.txt' %model_name
ckpt_name = 'best-%s.ckpt' %(model_name[14:-4])

hparam2epochs = {}
for hparam in '7 11 13 14 23 29 35 40 52 54'.split():
    hparam2epochs[int(hparam)] = 30
for hparam in '1 2 9 12 18 21 26 32 33 38 42 43 48 50'.split():
    hparam2epochs[int(hparam)] = 20
for hparam in range(55):
    if not hparam in hparam2epochs:
        hparam2epochs[hparam] = 10

key2ranks = {}
total = 0.0
count = 0
for run in '1-1 1-2 1-3 2-1 2-2 2-3 3-1 3-2 3-3'.split():
    for what in 'fso':
        model_dir_prefix = 'c-' + what + '-' + run
        found = []
        for entry in os.listdir(opt_data_dir):
            if not entry.startswith(model_dir_prefix):
                continue
            if opt_verbose: print('checking', entry)
            model_path = os.path.join(opt_data_dir, entry, ckpt_name)
            log_path = os.path.join(opt_data_dir, entry, log_name)
            if not os.path.exists(log_path):
                if opt_verbose: print('\tlog file not found')
                continue
            hparam = int(entry.split('-')[-1])
            if opt_epochs and opt_epochs != hparam2epochs[hparam]:
                if opt_verbose: print('\tnumber of epochs not selected')
                continue
            f = open(log_path, 'rt')
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('Best validation set accuracy:'):
                    line = line.replace('(', ' ')
                    line = line.replace(',', ' ')
                    fields = line.split()
                    # [0]  [1]        [2] [3]       [4]    [5]     [6]
                    # Best validation set accuracy: tensor(0.8536, device='cuda:0')
                    assert len(fields) >= 7
                    assert fields[4] == 'tensor'
                    assert fields[6].startswith('device=')
                    score = float(fields[5])
                    tie_breaker = hashlib.sha256(b'%d:%s:%s' %(
                        len(opt_seed), opt_seed, model_path.encode('utf-8')
                    )).hexdigest()
                    found.append((-score, tie_breaker, model_path, hparam))
                    if opt_verbose: print('\tdetected score', score, 'for', hparam, tie_breaker)
                    break
            f.close()
        assert found
        found.sort()
        rank = 1.0
        for nscore, tie_breaker, model_path, hparam in found:
            key = (what, hparam)
            if not key in key2ranks:
                key2ranks[key] = []
            #key2ranks[key].append((rank, nscore, tie_breaker, model_path))
            key2ranks[key].append(rank)
            rank += 1.0
            if opt_verbose:
                # also get average score
                total -= nscore
                count += 1

for hparam in range(1,55):
    row = []
    row.append('%d' %hparam)
    ranks = []
    for what in 'fso':
        key = (what, hparam)
        try:
            avg_rank = sum(key2ranks[key]) / float(len(key2ranks[key]))
        except KeyError:
            avg_rank = 0.0
        ranks.append((avg_rank, what))
    ranks.sort()
    row.append('%.1f' %(ranks[0][0]))
    row.append('%.1f' %(ranks[-1][0]))
    row.append('%s' %(ranks[0][1]))
    row.append('%s' %(ranks[-1][1]))
    print('\t'.join(row))

if opt_verbose:
    print('average score', 100.0*total / float(count))

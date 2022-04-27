#!/usr/bin/env python

import os
import sys

model_dir_prefix = sys.argv[1]  # e.g. c-f-3-3 (without -${HPARAM})
model_name       = sys.argv[2]  # e.g. training-with-sea-aio

log_name = 'stdout-%s.txt' %model_name

example = """
Best validation set accuracy: tensor(0.8536, device='cuda:0')
"""

found = []
for entry in os.listdir():
    if not entry.startswith(model_dir_prefix):
        continue
    print('checking', entry)
    model_path = os.path.join(entry, 'best.ckpt')
    if not os.path.exists(model_path):
        print('\tmodel file not found: previously deleted or not ready yet --> skipping folder')
        continue
    log_path = os.path.join(entry, log_name)
    if not os.path.exists(log_path):
        print('\tlog file not found')
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
            print('\tdetected score', score)
            found.append((-score, model_path))
            break
    f.close()

assert found
found.sort()
print('Keeping', found[0][1])
print('Deleting:')
for _, model_path in found[1:]:
    print('\t%s' %model_path)

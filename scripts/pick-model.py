#!/usr/bin/env python

import hashlib
import os
import random
import sys

def usage():
    print('Usage: $0 [options] model-dir-prefix model-name')
    # TODO: print more details how to use this script

opt_seed = b'101'
opt_dry_run = False
opt_verbose = False
while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h', '-n'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--seed':
        opt_seed = sys.argv[1].encode('utf-8')
        del sys.argv[1]
        if not opt_seed:
            # use system randomness as non-deterministic seed
            opt_seed = b'%064x' %random.getrandbits(256)
    elif option == '--verbose':
        opt_verbose = True
    elif option in ('-n', '--dry-run'):
        opt_dry_run = True
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

model_dir_prefix = sys.argv[1]  # e.g. c-f-3-3 (without -${HPARAM})
model_name       = sys.argv[2]  # e.g. training-with-sea-aio

assert model_name.startswith('training-with-')
assert model_name.endswith('-aio')

log_name = 'stdout-%s.txt' %model_name
ckpt_name = 'best-%s.ckpt' %(model_name[14:-4])

example = """
Best validation set accuracy: tensor(0.8536, device='cuda:0')
"""

found = []
for entry in os.listdir():
    if not entry.startswith(model_dir_prefix):
        continue
    if opt_verbose: print('checking', entry)
    model_path = os.path.join(entry, ckpt_name)
    if not os.path.exists(model_path):
        if opt_verbose: print('\tmodel file not found: previously deleted or not ready yet --> skipping folder')
        continue
    log_path = os.path.join(entry, log_name)
    if not os.path.exists(log_path):
        if opt_verbose: print('\tlog file not found')
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
            if opt_verbose: print('\tdetected score', score)
            tie_breaker = hashlib.sha256(b'%d:%s:%s' %(
                len(opt_seed), opt_seed, model_path.encode('utf-8')
            )).hexdigest()
            found.append((-score, tie_breaker, model_path))
            break
    f.close()

assert found  # script is only supposed to be run after a model has been trained successfully
found.sort()
print('Keeping', found[0][-1], 'with score', -found[0][0])
del found[0]
found.sort(key=lambda item: item[-1])
if opt_dry_run:
    print('Would delete if not in dry-run mode:')
elif found:
    print('Deleting:')
else:
    print('Nothing to delete')
for _, _, model_path in found:
    print('\t%s' %model_path)
    if not opt_dry_run:
        os.unlink(model_path)

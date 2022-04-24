#!/usr/bin/env python

import random
random.seed(101)

neps = []
seed = 1
for lr1 in (2,10,46):
    for lr2 in (6, 30, 139):
        for fre in (0, 3):
            for vbs in (16, 64, 256):
                if not neps:
                    for _ in range(6):
                        neps.append(10)
                    for _ in range(3):
                        neps.append(20)
                    for _ in range(2):
                        neps.append(30)
                    random.shuffle(neps)
                print(seed, lr1, lr2, fre, vbs, neps[0])
                del neps[0]
                seed += 1


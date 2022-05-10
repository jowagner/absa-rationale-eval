#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import math
import random
random.seed(101)

samples = 100
resolution = 640
min_distance = 0.35
max_attempts = 20

parameters = [
        ('LR1',   1,  100, 'exp', 'int', None),
        ('LR2',   3,  300, 'exp', 'int', None),
        ('FRZ',   0,    7, 'lin', 'int', None),
        ('VBS',   8,  512, 'exp', 'int', 8),
        ('HL1', 384, 3072, 'exp', 'int', 64),
        ('HL2',  64,  512, 'exp', 'int', 16),
        ('DP1', 0.1,  0.8, 'lin', 'fp2', None),
        ('DP2', 0.1,  0.8, 'lin', 'fp2', None),
        ('DP3', 0.1,  0.8, 'lin', 'fp2', None),
        ('EPO',   8,   25, 'exp', 'int', None),
]

def format_hparams(hparams):
    row = []
    global parameters
    index = 0
    for _, _, _, _, dtype, _ in parameters:
        value = hparams[index]
        if dtype == 'int':
            row.append('%4d' %value)
        elif dtype == 'fp2':
            row.append('%.2f' %value)
        else:
            raise NotImplementedError
        index += 1
    return ' '.join(row)

picked = set()

def pick_candidate():
    global parameters
    global resolution
    ivres = 1.0 / resolution
    hparams = []
    coords  = []
    for _, start, end, distr, dtype, rounding in parameters:
        if distr == 'exp':
            if dtype == 'int':
                b = (end / float(start)) ** ivres
                if rounding:
                    value = rounding * int(0.5 +
                            (start * b ** random.randrange(resolution+1)) / rounding)
                else:
                    value = int(0.5 + start * b ** random.randrange(resolution+1))
                coord = ivres * math.log(value/float(start)) / math.log(b)
            else:
                raise NotImplementedError
        elif distr == 'lin':
            if rounding:
                raise NotImplementedError
            if dtype == 'int':
                value = random.randrange(start, end+1)
                coord = (value-start) / float(end-start)
            elif dtype == 'fp2':
                startx100 = int(0.5+start*100)
                endx100 = int(0.5+end*100)
                valuex100 = random.randrange(startx100, endx100+1)
                value = 0.01 * valuex100
                coord = (valuex100-startx100) / float(endx100-startx100)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        hparams.append(value)
        coords.append(coord)
    return tuple(hparams), tuple(coords)

def get_nn_distance(coords, picked):
    nearest = None
    for ref_coords in picked:
        square_distance = 0.0
        for index, coord in enumerate(coords):
            square_distance += (coord - ref_coords[index]) ** 2
        if nearest is None:
            nearest = square_distance
        elif square_distance < nearest:
            nearest = square_distance
    return nearest ** 0.5

def pick():
    global min_distance
    global max_attempts
    global picked
    best_candidate  = None
    best_cube_coord = None
    best_distance   = 0.0
    attempts = 0
    #print()
    for _ in range(min(max_attempts, 5+len(picked)**2)):
        candidate, cube_coord = pick_candidate()
        attempts += 1
        #print('candidate:', candidate, cube_coord)
        if not picked:
            best_candidate = candidate
            best_cube_coord = cube_coord
            break
        nn_distance = get_nn_distance(cube_coord, picked)
        #print('d =', nn_distance)
        if nn_distance > min_distance:
            best_candidate = candidate
            best_cube_coord = cube_coord
            best_distance  = nn_distance
            break
        if best_candidate is None:
            best_candidate = candidate
            best_cube_coord = cube_coord
            best_distance  = nn_distance
        elif nn_distance > best_distance:
            best_candidate = candidate
            best_cube_coord = cube_coord
            best_distance  = nn_distance
    picked.add(cube_coord)
    #print()
    return (best_candidate, best_distance, attempts)

neps = []
for seed in range(1, samples + 1):
    hparams, d, attempts = pick()
    hparams = format_hparams(hparams)
    if not neps:
        for _ in range(6):
            neps.append(10)
        for _ in range(3):
            neps.append(20)
        for _ in range(2):
            neps.append(30)
        random.shuffle(neps)
    print(1000+seed, hparams, neps[0], attempts, d)
    del neps[0]
    seed += 1


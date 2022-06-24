#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# module initialisation

function_words = set()

# public functions

def init_function_words(data_prefix = 'data/'):
    global function_words
    with open(data_prefix + 'function-words.txt', 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            function_words.add(line.split()[0])
    return function_words

def get_confusion_matrix(sea, rationale, raw_tokens, exclude_function_words = True):
    global function_words
    tn, fp, fn, tp = 0, 0, 0, 0
    for index, annotation in enumerate(sea):
        if exclude_function_words \
        and raw_tokens[index].lower() in function_words:
            # exclude this token from the evaluation
            continue
        if annotation == 'I' and index in rationale:
            tp += 1
        elif annotation == 'I':
            fn += 1
        elif annotation == 'O' and index in rationale:
            fp += 1
        elif annotation == 'O':
            tn += 1
        else:
            raise ValueError('Unsupported SEA annotation %r at index %d' %(annotation, index))
    return tn, fp, fn, tp

def get_fscore(confusion):
    tn, fp, fn, tp = confusion
    try:
        p = tp / float(tp+fp)
    except ZeroDivisionError:
        p = 1.0
    try:
        r = tp / float(tp+fn)
    except ZeroDivisionError:
        r = 1.0
    try:
        f = 2.0 * p * r / (p+r)
    except ZeroDivisionError:
        f = 0.0
    return f

class FscoreSummaryTable:

    def __init__(selfi, n_thresholds = 1001):
        self.table = {}
        self.n_thresholds = n_thresholds
        for threshold in range(n_thresholds):
            d = []
            for _ in range(4):
                d.append(0)
            for _ in range(12):
                d.append(0.0)
            d.append(0)
            d.append(0)
            self.table[threshold] = d

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        if type(key) == int:
            raise KeyError('FscoreSummaryTable rejected attempt to overwrite table row')
        self.table[key] = value

    def update(self, seq_length, data):
        ''' update summary stats for thresholds in steps of 0.001
            (using integer operations to avoid numerical issues)
        '''
        for threshold in range(1001):
            d = self[threshold]
            r_length = (seq_length * threshold + 500) // 1000
            row = data[r_length]
            for k in range(8):
                d[k] += row[k]
            for k in range(4):
                d[8+k]  += (row[k] / float(seq_length))
                d[12+k] += (row[4+k] * float(seq_length))
            d[16] += seq_length
            d[17] += 1

def main():
    import sys
    confusion_matrix = map(int, sys.argv[1:5])
    sys.stdout.write('%.4f\n' %get_fscore(confusion_matrix))

if __name__ == "__main__":
    main()


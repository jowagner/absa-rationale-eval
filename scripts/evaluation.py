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

def main():
    import sys
    confusion_matrix = map(int, sys.argv[1:5])
    sys.stdout.write('%.4f\n' %get_fscore(confusion_matrix))

if __name__ == "__main__":
    main()


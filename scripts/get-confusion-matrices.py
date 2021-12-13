#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys

def new_confusion_matrix():
    retval = {}
    for gold in 'negative neutral positive'.split():
        for pred in 'negative neutral positive'.split():
            retval[(gold, pred)] = 0
    return retval

tables = {}

for tr_type_short, tr_type_long in [
        ('f', 'Tr=1-Full'),
        ('s', 'Tr=2-SE'),
        ('o', 'Tr=3-Other'),
        ('a', 'Tr=4-Concat'),
]:
    set_index = 0
    for set_major, set_minor in [
            (1,1), (1,2), (1,3),
            (2,1), (2,2), (2,3),
            (3,1), (3,2), (3,3),
    ]:
        set_s = 'classifier-%d' %(set_index+1)
        any_s = 'all-classifiers'
        f = open('c-%s-%s-%s/saliency-xfw-stdout.txt' %(
            tr_type_short, set_major, set_minor
        ), 'rt')
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith('=== Item ==='):
                continue
            assert f.readline().isspace()
            header = f.readline().split()
            fields = f.readline().split()
            # seed    question        set     index   domain  mask    gold    pred    tokens  subwords        SEA-I   SEA-O   SEA-percentage
            if fields[header.index('set')] != 'test':
                continue
            domain = fields[header.index('domain')]
            mask   = fields[header.index('mask')]
            gold   = fields[header.index('gold')]
            pred   = fields[header.index('pred')]
            sea_i_tags = int(fields[header.index('SEA-I')])
            sea_o_tags = int(fields[header.index('SEA-O')])
            assert sea_i_tags >= 0  # tag counts must not be negative
            assert sea_o_tags >= 0
            assert sea_i_tags + sea_o_tags > 0  # must have at least 1 token
            if sea_i_tags == 0:
                sea = 'no-I'
            elif sea_o_tags == 0:
                sea = 'no-O'       # above assertion excludes no-O-and-no-I case
            else:
                sea = 'both-I-and-O'
            if mask == 'None':
                te_type = 'Te=Full (Mask=None)'
            elif mask == 'O':
                te_type = 'Te=SE (Mask=O)'
            elif mask == 'I':
                te_type = 'Te=Z-Other (Mask=I)'
            else:
                raise ValueError('Mask %r in %r' %(mask, line))
            for cm_key in [
                    (tr_type_long, te_type, any_s, domain,       'any-SEA'),
                    (tr_type_long, te_type, any_s, 'any-domain', 'any-SEA'),
                    (tr_type_long, te_type, any_s, domain,       sea),
                    (tr_type_long, te_type, any_s, 'any-domain', sea),
                    (tr_type_long, te_type, set_s, domain,       'any-SEA'),
                    (tr_type_long, te_type, set_s, 'any-domain', 'any-SEA'),
                    (tr_type_long, te_type, set_s, domain,       sea),
                    (tr_type_long, te_type, set_s, 'any-domain', sea),
            ]:
                if cm_key not in tables:
                    tables[cm_key] = new_confusion_matrix()
                tables[cm_key][(gold, pred)] += 1
        f.close()
        set_index += 1


for cm_key in tables.keys():
    section = []
    row = ['gold \\ prediction', 'Neg', 'Neu', 'Pos', 'Total']
    section.append(row)
    total = 0
    col_totals = {}
    correct = 0
    for pred in 'negative neutral positive'.split():
        col_totals[pred] = 0
    for gold in 'negative neutral positive'.split():
        row = []
        row.append(gold)
        row_total = 0
        for pred in 'negative neutral positive'.split():
            count = tables[cm_key][(gold, pred)]
            row.append('%d' %count)
            row_total += count
            col_totals[pred] += count
            total += count
            if gold == pred:
                correct += count
        row.append('%d' %row_total)
        section.append(row)
    row = []
    row.append('total')
    for pred in 'negative neutral positive'.split():
        count = col_totals[pred]
        row.append('%d' %count)
    row.append('%d' %total)
    #row.append('accuracy')
    row.append('%.9f' %(100.0*correct/float(total)))
    section.append(row)
    section.append([])
    row_index = 0
    for row in section:
        sys.stdout.write('\t'.join(cm_key))
        sys.stdout.write('\t%d\t' %row_index)
        sys.stdout.write('\t'.join(row))
        sys.stdout.write('\n')
        row_index += 1

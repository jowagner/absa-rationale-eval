#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

import hashlib
import lime
import numpy as np
import os
import sys
import time

import raw_data

max_n_features = int(sys.argv[1])
dataset_index  = int(sys.argv[2])
prob_dir = 'tasks/probs'
task_dir = 'tasks'

num_samples = 10000
mask_string = '[MASK]'

# TODO: command line options to set above variables

# Fetching data, training a classifier

dataset_name = ['trainig', 'test'][dataset_index]
dataset = raw_data.get_dataset(raw_data.get_filenames(), dataset_index)[0]

print('loaded %s data with %d items' %(dataset_name, len(dataset)))

class_names = 'negative neutral positive'.split()
print('class_names:', class_names)

# Explaining predictions using lime

from lime import lime_text
from lime.lime_text import LimeTextExplainer

def my_tokeniser(text):
    return text.split()

explainer = LimeTextExplainer(
    class_names      = class_names,
    split_expression = my_tokeniser,
    bow              = False,
    mask_string      = mask_string,
    random_state     = 101,
)

# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.
# Below, we generate explanations for labels 0, 1 and 2.

item_info = None   # global variable as LIME does not provide a way to pass along item information
item_index = None

def get_package_name(items):
    global item_info
    h = hashlib.sha256()
    h.update(item_info.encode('UTF-8') + b'\n')
    for item, mask in items:
        h.update(('%d:%s\n' %(len(item), ' '.join(item))).encode('UTF-8'))
    return '%d-%s' %(len(items), h.hexdigest())

def add_probs(mask2triplet, prob_path):
    name = prob_path.split('/')[-1]
    n_items = int(name.split('-')[0])
    f = open(prob_path, 'rt')
    for row_index in range(n_items):
        fields = f.readline().split()
        assert len(fields) == 4
        mask = int(fields[0])
        triplet = []
        for col_index in (1,2,3):
            triplet.append(float(fields[col_index]))
        mask2triplet[mask] = triplet
    if f.readline():
        print('Warning: trailing line(s) in', prob_path)
    f.close()
    return mask2triplet

def get_cache():
    global prob_dir
    global item_info
    global dataset_index
    global item_index
    item_dir = '%s/%d/%d' %(prob_dir, dataset_index, item_index)
    retval = {}
    if not os.path.exists(item_dir):
        return retval
    for entry in os.listdir(item_dir):
        candidate_path = os.path.join(item_dir, entry)
        if '-' in entry and os.path.isfile(candidate_path):
            add_probs(retval, candidate_path)
    return retval

def write_package(package, name):
    global item_info
    global dataset_index
    global item_index
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    f = open(task_dir + '/' + name + '.new', 'wt')
    for item, mask in package:
        assert '\n' not in item
        f.write('%d\t%d\t%s\t%d\t' %(dataset_index, item_index, item_info, mask))
        f.write(item)
        f.write('\n')
    f.close()

def get_missing_predictions(items):
    ''' input: list of (item, mask) pairs
        output: dictionary mapping each mask to its probability
                triplet
    '''
    print('call of get_missing_predictions() with batch size', len(items))
    if not items:
        return {}
    assert type(items[0]) is tuple
    assert len(items[0]) == 2
    # write task files
    missing = []
    concurrent = False
    while items:
        if len(items) > 2097152:
            pick = 1048576
        elif len(items) > 1048576:
            pick = int(len(items) // 2)
        else:
            pick = len(items)
        package = items[:pick]
        items = items[pick:]
        name = get_package_name(package)
        prob_path = '%s/%d/%d/%s' %(prob_dir, dataset_index, item_index, name)
        if os.path.exists(task_dir + '/' + name + '.new'):
            print('Warning: detected concurrent run in same folder (1)\n')
            concurrent = True
        elif os.path.exists(prob_path):
            print('Warning: detected concurrent run in same folder (2)\n')
            concurrent = True
        else:
            # normal case: task file needs to be written
            write_package(package, name)
        missing.append(prob_path)
    print('Waiting for predictions for %d package(s)...\n' %len(missing))
    # collect answers
    retval = {}
    step = 0
    next_wait = 0.25
    while missing:
        m_index = step % len(missing)
        prob_path = missing[m_index]
        if os.path.exists(prob_path):
            if concurrent:
                # increase chances the file is complete
                time.sleep(2.0)
            # get_probs(prob_path)))
            add_probs(retval, prob_path)
            del missing[m_index]
            next_wait = 0.25
        if missing:
            time.sleep(next_wait)
        next_wait = min(15.0, 2.0 * next_wait)
        step += 1
    return retval

def get_mask(sentence):
    global mask_string
    tokens = my_tokeniser(sentence)
    mask = 0
    for token in tokens:
        mask <<= 1
        if token == mask_string:
            mask = mask + 1
    return mask

def my_predict_proba(items):
    ''' input: list of d strings
        output: a (d, 3) numpy array with probabilities for
                negative, neutral and positive
    '''
    print('call of predict_proba() with batch size', len(items))
    assert type(items[0]) == str
    # deduplicate and find cached items
    # (for short sequences, LIME asks for predictions for the
    # same input over and over again; furthermore, caching speeds
    # up repeat runs, e.g. increasing the number of samples)
    row2mask = []
    cache = get_cache()
    masks = set()
    new = []
    for item in items:
        mask = get_mask(item)
        row2mask.append(mask)
        if mask not in masks and mask not in cache:
            new.append((item, mask))
            masks.add(mask)
    # get missing predictions
    mask2triplet = get_missing_predictions(new)
    # assemble probability triplets
    retval = np.zeros(shape = (len(items), 3), dtype = np.float64)
    for row_index, mask in enumerate(row2mask):
        try:
            triplet = mask2triplet[mask]
        except KeyError:
            triplet = cache[mask]
        for col_index in range(3):
            retval[row_index, col_index] = triplet[col_index]
    return retval

    # TODO: Can we return one-hot vectors or does LIME not work well for
    #       over-confident classifiers?
    #       https://github.com/marcotcr/lime/issues/615 suggests probabilities are much preferred.

for item_index, item in enumerate(dataset):
    domain, opinion_id, text, \
            tokens, sea, \
            entity_type, attribute_label, \
            target, span, \
            polarity = item
    print()
    print('== item index %d ==' %item_index)
    print()
    print('domain', domain)
    print('opinion_id', opinion_id)
    print('text', text)
    print('tokens', tokens)
    print('entity_type', entity_type)
    print('attribute_label', attribute_label)
    print('target', target)
    print('span', span)
    print('polarity', polarity)
    print()

    # TODO: Seed LIME before every instance so that changing num_samples does not prohibit
    #       re-use of predictions.
    #       Actually, increasing num_samples seems to completely change the permutations.
    #       The first previous_num_samples items are not the same.

    assert '\t' not in entity_type
    assert '\t' not in attribute_label
    item_info = '%s\t%s\t%s' %(domain, entity_type, attribute_label)  # hack to pass info around LIME

    exp = explainer.explain_instance(
        text_instance   = ' '.join(tokens),  # raw text (before my_tokeniser())
        classifier_fn   = my_predict_proba,
        num_features    = max_n_features,
        num_samples     = num_samples,
        labels          = [0,1,2],
    )

    print()
    for class_index in range(3):
        print ('Explanation for class [%d] = %s' %(class_index, class_names[class_index]))
        items = list(map(str, exp.as_list(label=class_index)))
        for item in items[:10]:
            print(item)
        print()

    for index, centre in enumerate(tokens):
        scores = []
        for class_index in range(3):
            scores.append(sorted(exp.as_map()[class_index])[index][1])
        left_context = 8 * ['[>>>]']
        left_context = left_context + tokens[max(0, index-20):index]
        left_context = ' '.join(left_context)
        left_context = left_context[-40:]
        centre = ' %s ' %centre
        if len(centre) > 15:
            centre = centre[:6] + '...' + centre[-6:]
        right_context = tokens[index+1:index+20] + 8 * ['[<<<]']
        right_context = ' '.join(right_context)
        right_context = right_context[:40]
        print('[%3d] %9.6f %9.6f %9.6f %s %15s %s' %(index, scores[0], scores[1], scores[2], left_context, centre, right_context))

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
    mask_string      = '[MASK]',
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
    for item in items:
        h.update(('%d:%s\n' %(len(item), ' '.join(item))).encode('UTF-8'))
    return '%d-%s' %(len(items), h.hexdigest())

def get_probs(prob_path):
    name = prob_path.split('/')[-1]
    n_items = int(name.split('-')[0])
    retval = np.zeros(shape = (n_items, 3), dtype = np.float64)
    f = open(prob_path, 'rt')
    for row_index in range(n_items):
        fields = f.readline().split()
        assert len(fields) == 3
        for col_index in (0,1,2):
            retval[row_index, col_index] = float(fields[col_index])
    f.close()
    return retval

def write_package(package, name):
    global item_info
    global dataset_index
    global item_index
    item_task_dir = '%s/%d/%d' %(task_dir, dataset_index, item_index)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    f = open(task_dir + '/' + name + '.new', 'wt')
    for item in package:
        assert '\n' not in item
        f.write('%d\t%d\t%s\t' %(dataset_index, item_index, item_info))
        f.write(item)
        f.write('\n')
    f.close()

def my_predict_proba(items):
    ''' input: list of d strings
        output: a (d, k) numpy array with probabilities of k classes
                (classifier.predict_proba for ScikitClassifiers)
    '''
    print('call of predict_proba() with batch size', len(items))
    assert type(items[0]) == str
    # TODO: check for masks already tried for this item
    #       (package-level caching implemented so far does
    #       not work with LIME as changing the set size
    #       completely changes the order of permutations)

    # write task files
    n_packages = 0
    found = []
    missing = []
    while items:
        if len(items) > 2097152:
            pick = 1048576
        elif len(items) > 1048576:
            pick = int(len(items) // 2)
        else:
            pick = len(items)
        n_packages += 1
        package = items[:pick]
        items = items[pick:]
        name = get_package_name(package)
        # TODO: should we test for pre-existing task file?
        prob_path = '%s/%d/%d/%s' %(prob_dir, dataset_index, item_index, name)
        if os.path.exists(prob_path):
            found.append((n_packages, get_probs(prob_path)))
        else:
            write_package(package, name)
            missing.append((n_packages, prob_path))
    # collect answers
    step = 0
    next_wait = 0.25
    while missing:
        m_index = step % len(missing)
        package_rank, prob_path = missing[m_index]
        if os.path.exists(prob_path):
            found.append((package_rank, get_probs(prob_path)))
            del missing[m_index]
            next_wait = 0.25
        time.sleep(next_wait)
        next_wait = min(15.0, 2.0 * next_wait)
        step += 1
    # combine answers
    found.sort()
    parts = lambda x: x[1], found
    return np.concatenate(parts, axis = 0)

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
        print ('Explanation for class [%d] = %s' %(class_index, ng_class_names[class_index]))
        items = list(map(str, exp.as_list(label=class_index)))
        for item in items[:10]:
            print(item)
        print()

    tokens = newsgroups_test.data[idx].split()
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

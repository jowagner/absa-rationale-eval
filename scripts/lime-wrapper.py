#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

import bz2
import hashlib
import lime
import numpy as np
import os
import sys
import time

import raw_data

def usage():
    print('Usage: $0 [options] [preload]>')
    # TODO: print more details how to use this script

max_n_features = 9999
dataset_index  = 0  # 0 = training set, 1 = test set
max_tasks = None
preload_tasks = False
prob_dir = 'tasks/probs'
task_dir = 'tasks'
num_samples = 10000
mask_string = '[MASK]'
opt_verbose = False
opt_abort_on_cache_miss = False

while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--verbose':
        opt_verbose = True
    elif option == '--abort-on-cache-miss':
        opt_abort_on_cache_miss = True
    elif option in ('--train', '--for-training-set'):
        dataset_index = 0
    elif option in ('--test', '--for-test-set'):
        dataset_index = 1
    elif option in ('--preload', '--do-not-wait-for-predictions'):
        preload_tasks = True
    elif option in ('--tasks', '--max-tasks'):
        max_tasks = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--samples', '--num-samples'):
        num_samples = int(sys.argv[1])
        del sys.argv[1]
    # TODO: add more options
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

if preload_tasks and max_tasks is None:
    max_tasks = 100

# Fetching data, training a classifier

dataset_name = ['trainig', 'test'][dataset_index]
dataset = raw_data.get_dataset(raw_data.get_filenames(), dataset_index)[0]

if opt_verbose: print('# Loaded %s data with %d items' %(dataset_name, len(dataset)))

class_names = 'negative neutral positive'.split()
if opt_verbose: print('# Class_names:', class_names)

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
        print('# Warning: trailing line(s) in', prob_path)
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
    global max_tasks
    if max_tasks is not None:
        if max_tasks == 0:
            print('# Reached maximum number of tasks')
            sys.exit(0)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    path_final = task_dir + '/' + name + '.new'
    path_partial = path_final + '.part'
    f = bz2.open(path_partial, 'wt')
    for item, mask in package:
        assert '\n' not in item
        f.write('%d\t%d\t%s\t%d\t' %(dataset_index, item_index, item_info, mask))
        f.write(item)
        f.write('\n')
    f.close()
    os.rename(path_partial, path_final)
    if max_tasks is not None:
        max_tasks -= 1

def is_in_preparation(dataset_index, item_index, name):
    if os.path.exists(task_dir):
        found_task = False
        for entry in os.listdir(task_dir):
            if entry.startswith(name):
                # .new / .new.part / .$PID-$HOSTNAME suffix
                found_task = True
                break
        if found_task:
            if opt_verbose: print('# Warning: detected concurrent %s for %s in task folder' %(entry, name))
            return True
    prob_item_dir = '%s/%d/%d' %(prob_dir, dataset_index, item_index)
    if os.path.exists(prob_item_dir):
        found_probs = False
        for entry in os.listdir(prob_item_dir):
            if entry.startswith(name):
                # '' / .part suffix
                found_probs = True
        if found_probs:
            if opt_verbose: print('# Warning: detected concurrent %s for %s in prediction folder %s' %(entry, name, prob_item_dir))
            return True
    return False

def get_missing_predictions(items):
    ''' input: list of (item, mask) pairs
        output: dictionary mapping each mask to its probability
                triplet
    '''
    global preload_tasks
    if opt_verbose: print('# Call of get_missing_predictions() with %d item(s)' %len(items))
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
        if is_in_preparation(dataset_index, item_index, name):
            concurrent = True
        else:
            # normal case: task file needs to be written
            write_package(package, name)
        missing.append(prob_path)
    if preload_tasks:
        return None
    if opt_verbose: print('# Waiting for predictions for %d package(s)...\n' %len(missing))
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

cache = None

def my_predict_proba(items):
    ''' input: list of d strings
        output: a (d, 3) numpy array with probabilities for
                negative, neutral and positive
    '''
    global cache
    total = len(items)
    if opt_verbose: print('# Call of predict_proba() with %d item(s)' %total)
    assert total > 0
    assert type(items[0]) == str
    # deduplicate and find cached items
    # (for short sequences, LIME asks for predictions for the
    # same input over and over again; furthermore, caching speeds
    # up repeat runs, e.g. increasing the number of samples)
    row2mask = []
    cache = get_cache()
    masks = set()
    new = []
    after_dedup = 0
    cache_miss = 0
    for item in items:
        mask = get_mask(item)
        row2mask.append(mask)
        if mask not in masks:
            after_dedup += 1
            if mask not in cache:
                cache_miss += 1
                new.append((item, mask))
                masks.add(mask)
    if opt_verbose: print('# %d item(s) after deduplication' %after_dedup)
    if opt_verbose: print('# %d unique items (%.1f%% raw, %.1f%% dedup) not in cache' %(
        cache_miss,
        100.0*cache_miss/float(total),
        100.0*cache_miss/float(after_dedup),
    ))
    if cache_miss and opt_abort_on_cache_miss:
        print('# aborting as --abort-on-cache-miss was specified')
        sys.exit(1)
    # get missing predictions
    mask2triplet = get_missing_predictions(new)
    # assemble probability triplets
    retval = np.zeros(shape = (len(items), 3), dtype = np.float64)
    for row_index, mask in enumerate(row2mask):
        try:
            triplet = mask2triplet[mask]
        except KeyError:
            triplet = cache[mask]
        except TypeError:
            if not preload_tasks:
                raise TypeError
            continue
        for col_index in range(3):
            retval[row_index, col_index] = triplet[col_index]
    return retval

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
    print('tokens', ' '.join(tokens))
    print('sea', ' '.join(sea))
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
        text_instance   = ' '.join(tokens),  # will be sent to my_tokeniser()
        classifier_fn   = my_predict_proba,
        num_features    = max_n_features,
        num_samples     = num_samples,
        labels          = [0,1,2],
    )

    try:
        triplet = cache[0]
    except KeyError:
        triplet = None
    if triplet:
        print()
        candidates = []
        for p_index, prob in enumerate(triplet):
            label = class_names[p_index]
            tiebreaker = '%d:%d:%s' %(dataset_index, item_index, label)
            tiebreaker = hashlib.sha256(tiebreaker.encode('UTF-8')).hexdigest()
            candidates.append((-prob, tiebreaker, label))
        candidates.sort()
        row = []
        row.append('Prediction:')
        row.append(candidates[0][-1])
        for prob in triplet:
            row.append('%.9f' %prob)
        print('\t'.join(row))

    if triplet or opt_verbose:
        print()

    if opt_verbose:
        expl_for_classes = range(3)
    else:
        expl_for_classes = []
    for class_index in expl_for_classes:
        print ('Explanation for class [%d] = %s' %(class_index, class_names[class_index]))
        items = list(map(str, exp.as_list(label=class_index)))
        for item in items[:10]:
            print(item)
        print()

    print('Scores:')
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
        row = []
        row.append('%6d\t%14.9f\t%14.9f\t%14.9f' %(index, scores[0], scores[1], scores[2]))
        if opt_verbose:
            row.append('%s %15s %s' %(left_context, centre, right_context))
        print('\t'.join(row))

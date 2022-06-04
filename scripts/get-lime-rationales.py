#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

import lime
import numpy as np
import sys

import raw_data

max_n_features = int(sys.argv[1])
dataset_index  = int(sys.argv[2])

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

# Previously, we used the default parameter for label when generating explanation, which works well in the binary case.
# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.
# Below, we generate explanations for labels 0 and 17.

def my_predict_proba(items):
    ''' input: list of d strings
        output: a (d, k) numpy array with probabilities of k classes
                (classifier.predict_proba for ScikitClassifiers)
    '''
    print('call of predict_proba() with batch size', len(items))
    assert type(items[0]) == str
    raise NotImplementedError
    # TODO: Can we return one-hot vectors or does LIME not work well for
    #       over-confident classifiers?

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

    assert '/' not in entity_type
    assert '/' not in attribute_label
    tokens.append('%s/%s/%s' %(domain, entity_type, attribute_label))

    exp = explainer.explain_instance(
        text_instance   = ' '.join(tokens),  # raw text (before my_tokeniser())
        classifier_fn   = my_predict_proba,
        num_features    = max_n_features,
        num_samples     = 10000,
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

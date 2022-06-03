#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sys
#from __future__ import print_function

max_n_features = int(sys.argv[1])

# Fetching data, training a classifier

# TODO: switch to ABSA data and classifiers

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# making class names shorter
ng_class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
ng_class_names[3] = 'pc.hardware'
ng_class_names[4] = 'mac.hardware'

print(','.join(ng_class_names))

# atheism,graphics,ms-windows.misc,pc.hardware,mac.hardware,x,misc.forsale,autos,motorcycles,baseball,hockey,crypt,electronics,med,space,christian,guns,mideast,politics.misc,religion.misc
#
# Again, let's use the tfidf vectorizer, commonly used for text.

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
ng_train_vectors = vectorizer.fit_transform(newsgroups_train.data)
ng_test_vectors = vectorizer.transform(newsgroups_test.data)

# This time we will use Multinomial Naive Bayes for classification, so that we can make reference to this document.

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=.01)
nb.fit(ng_train_vectors, newsgroups_train.target)

pred = nb.predict(ng_test_vectors)
score = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

print(score)
# 0.83501841939981736 # We see that this classifier achieves a very high F score. The sklearn guide to 20 newsgroups indicates that Multinomial Naive Bayes overfits this dataset by learning irrelevant stuff, such as headers, by looking at the features with highest coefficients for the model in general. We now use lime to explain individual predictions instead.


# Explaining predictions using lime

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, nb)

print(c.predict_proba([newsgroups_test.data[0]]).round(3))

#[[ 0.001  0.01   0.003  0.047  0.006  0.002  0.003  0.521  0.022  0.008
#   0.025  0.     0.331  0.003  0.006  0.     0.003  0.     0.001  0.009]]

from lime.lime_text import LimeTextExplainer

def my_tokeniser(text):
    return text.split()

explainer = LimeTextExplainer(
    class_names      = ng_class_names,
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
    global c
    print('call of predict_proba() with %d item(s)' %len(items))
    return c.predict_proba(items)

idx = 1340
print('data instance %d with %d tokens' %(idx, len(my_tokeniser(newsgroups_test.data[idx]))))

exp = explainer.explain_instance(
    text_instance   = newsgroups_test.data[idx],  # raw text (before my_tokeniser())
    classifier_fn   = my_predict_proba,
    num_features    = max_n_features,
    num_samples     = 10000,
    labels          = [0,1,2],
)
print('Document id: %d' % idx)
print('Predicted class =', ng_class_names[nb.predict(ng_test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % ng_class_names[newsgroups_test.target[idx]])

#   Document id: 1340
#   Predicted class = atheism
#   True class: atheism

#  Now, we can see the explanations for different labels. Notice that the positive and negative signs are with respect to a particular label - so that words that are negative towards class 0 may be positive towards class 15, and vice versa.

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

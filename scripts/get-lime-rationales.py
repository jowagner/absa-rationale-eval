#!/usr/bin/env python

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
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(','.join(class_names))

# atheism,graphics,ms-windows.misc,pc.hardware,mac.hardware,x,misc.forsale,autos,motorcycles,baseball,hockey,crypt,electronics,med,space,christian,guns,mideast,politics.misc,religion.misc
# 
# Again, let's use the tfidf vectorizer, commonly used for text.

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# This time we will use Multinomial Naive Bayes for classification, so that we can make reference to this document.

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)

pred = nb.predict(test_vectors)
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

explainer = LimeTextExplainer(class_names=class_names, split_expression=my_tokeniser)

# Previously, we used the default parameter for label when generating explanation, which works well in the binary case.
# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.
# Below, we generate explanations for labels 0 and 17.

def my_predict_proba(items):
    global c
    print('call of predict_proba() with %d item(s)' %len(items))
    return c.predict_proba(items)

idx = 1340
exp = explainer.explain_instance(newsgroups_test.data[idx], my_predict_proba, num_features=max_n_features, labels=[0,])
print('Document id: %d' % idx)
print('Predicted class =', class_names[nb.predict(test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

#   Document id: 1340
#   Predicted class = atheism
#   True class: atheism

#  Now, we can see the explanations for different labels. Notice that the positive and negative signs are with respect to a particular label - so that words that are negative towards class 0 may be positive towards class 15, and vice versa.

print ('Explanation for class %s' % class_names[0])
print ('\n'.join(map(str, exp.as_list(label=0))))

tokens = newsgroups_test.data[idx].split()

for index, score in sorted(exp.as_map()[0]):
    print(index, score, tokens[max(0, index-5):index], repr(tokens[index]), tokens[index+1:index+6])

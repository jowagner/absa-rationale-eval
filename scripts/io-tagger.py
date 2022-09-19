#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import numpy
import os
import random
from sklearn.tree import DecisionTreeClassifier
import sys

import evaluation

def usage():
    print('Usage: $0 [options] c-f-1-1')
    # TODO: print more details how to use this script

opt_workdir = sys.argv[1]
opt_folds   = int(sys.argv[2])
opt_leaf_size = int(sys.argv[3])
opt_seed_for_data_split = 101
opt_viz_tree = False

opt_training_data = os.path.join(opt_workdir, 'lime-features-tr.tsv')
opt_test_data     = os.path.join(opt_workdir, 'lime-features-te.tsv')

if opt_viz_tree:
    from dtreeviz.trees import dtreeviz
    max_depth = 6
else:
    max_depth = 20

class IODataset:

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_sklearn_data(self):
        rows = len(self.data)
        columns = self.sea_column
        features = numpy.zeros((rows, columns), dtype=numpy.float64)
        targets  = numpy.zeros(rows, dtype=numpy.int8)
        for row_index, item in enumerate(self):
            for col_index, value in enumerate(item[:self.sea_column]):
                features[row_index, col_index] = float(value)
            targets[row_index] = int(item[self.sea_column])
        return features, targets


class IODatasetSubset(IODataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.data    = indices
        self.sea_column = dataset.sea_column
        try:
            self.feature_names = dataset.feature_names
        except:
            pass

    def __getitem__(self, index):
        return self.dataset[self.data[index]]


class IODatasetGrouped(IODataset):

    def get_folds(self, n_folds):
        groups_by_size = []
        for group_key in self.groups2indices.keys():
            indices = self.groups2indices[group_key]
            groups_by_size.append((len(indices), group_key))
        groups_by_size.sort(reverse=True)
        # pad list to multiple of n_folds groups
        while len(groups_by_size) % n_folds > 0:
            groups_by_size.append((0, None))
        # process groups
        fold2indices = []
        for _ in range(n_folds):
            fold2indices.append([])
        start = 0
        while start < len(groups_by_size):
            selection = groups_by_size[start: start+n_folds]
            random.shuffle(selection)
            for fold_index in range(n_folds):
                n_items, group_key = selection[fold_index]
                if n_items > 0:
                    indices = self.groups2indices[group_key]
                    fold2indices[fold_index] += indices
            start += n_folds
        # assemble subsets
        retval = []
        for te_fold_index in range(n_folds):
            te_indices = fold2indices[te_fold_index]
            tr_indices = []
            for tr_fold in range(n_folds-1):
                tr_fold_index = (te_fold_index + 1 + tr_fold) % n_folds
                tr_indices += fold2indices[tr_fold_index]
            retval.append((
                IODatasetSubset(self, tr_indices),
                IODatasetSubset(self, te_indices),
            ))
        return retval

    
class IODatasetFromFile(IODatasetGrouped):

    def __init__(self, path):
        sent2indices = {}
        data = []
        f = open(path, 'rt')
        header = f.readline().strip('#').split()
        #print(header)
        id_column = header.index('item_ID')
        sea_column = header.index('SEA')
        n_items    = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('#') or line.isspace():
                continue
            fields = line.split()
            item_id = fields[id_column]
            sent_id = item_id[:item_id.rfind(':')]
            if sent_id not in sent2indices:
                sent2indices[sent_id] = []
            sent2indices[sent_id].append(n_items)
            data.append(fields)
            n_items += 1
        f.close()
        self.groups2indices = sent2indices
        self.data = data
        self.sea_column = sea_column
        self.feature_names = header[:sea_column]

overall_tr_dataset = IODatasetFromFile(opt_training_data)
overall_te_dataset = IODatasetFromFile(opt_test_data)

random.seed(opt_seed_for_data_split)


correct = 0
total   = 0
fold    = 0
for tr_dataset, te_dataset in overall_tr_dataset.get_folds(opt_folds):
    print('Fold with %d training items and %d test items' %(
        len(tr_dataset), len(te_dataset)
    ))
    # train model for this fold
    features, targets = tr_dataset.get_sklearn_data()
    model = DecisionTreeClassifier(
        max_depth = max_depth,
        min_samples_leaf  = opt_leaf_size,
        random_state = 101,
    )
    model.fit(features, targets)
    if opt_viz_tree:
        dtreeviz(model, features, targets,
                 target_name='SE-IO-tag',
                 feature_names = tr_dataset.feature_names,
                 class_names = 'O I'.split(),
       ).save('dt-%d-%d-%d-%d.svg' %(max_depth, opt_leaf_size, opt_folds, fold))
    te_features, te_gold_labels = te_dataset.get_sklearn_data()
    predictions = model.predict(te_features)
    for i in range(len(te_gold_labels)):
        if te_gold_labels[i] == predictions[i]:
            correct += 1
    total += len(te_gold_labels)
    fold += 1
    if opt_viz_tree:
        break
if not opt_viz_tree:
    print('Average accuracy: %.2f%%' %(100.0* correct / float(total)))



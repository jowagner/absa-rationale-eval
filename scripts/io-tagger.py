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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys

import evaluation

def usage():
    print('Usage: $0 [options] c-f-1-1')
    # TODO: print more details how to use this script

opt_workdir = sys.argv[1]
tagger_version = int(sys.argv[2])
opt_folds   = 20
opt_leaf_size = None  # will be set below depending on tagger version
opt_context   = None  # will be set below depending on tagger version
opt_seed_for_data_split = 101
opt_viz_tree = False
domains = 'laptop restaurant'.split()
labels  = 'O I'.split()

assert 1 <= tagger_version <= 4

tagger_version_to_default_leaf_size = {
    1: 1376,  # DT with per-token features
    2:  250,  # DT with context features   # TODO: tune
    3:   17,  # RF with per-token features
    4:   11,  # RF with context features   # TODO: tune
}

if opt_leaf_size is None:
    opt_leaf_size = tagger_version_to_default_leaf_size[tagger_version]

if opt_context is None:
    opt_context = 2 if tagger_version % 2 == 0 else 0

opt_training_data = os.path.join(opt_workdir, 'lime-features-tr.tsv')
opt_test_data     = os.path.join(opt_workdir, 'lime-features-te.tsv')

if opt_viz_tree:
    from dtreeviz.trees import dtreeviz
    max_depth = 6
else:
    max_depth = 20

# TODO: allow above setting to be changed from the command line


class IODataset:

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_sklearn_data(self, context = 0):
        rows = len(self.data)
        columns = self.sea_column
        if context > 0:
            token_specific_features = self.header.index('t_idx')
            id_column = self.header.index('item_ID')
            centre_features = columns
            columns += 4 * token_specific_features
        features = numpy.zeros((rows, columns), dtype=numpy.float64)
        targets  = numpy.zeros(rows, dtype=numpy.int8)
        for row_index, item in enumerate(self):
            for col_index, value in enumerate(item[:self.sea_column]):
                features[row_index, col_index] = float(value)
                if context > 0 and col_index < token_specific_features:
                    my_id = item[id_column]
                    block_index = 0
                    rel_pos = -context
                    while rel_pos <= context:
                        target_row_index = row_index + rel_pos
                        if 0 <= target_row_index < rows:
                            target_id = self[target_row_index][id_column]
                            if my_id == target_id:
                                target_col_index = centre_features + \
                                    block_index * token_specific_features + \
                                    col_index
                                features[target_row_index, target_col_index] = float(value)
                            # else: keep zero value
                        block_index += 1
                        rel_pos += 1
                        if rel_pos == 0:
                            rel_pos += 1
            targets[row_index] = int(item[self.sea_column])
        return features, targets


class IODatasetSubset(IODataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.data    = indices
        self.header  = dataset.header
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
        self.header = header
        self.sea_column = sea_column
        self.feature_names = header[:sea_column]

def mhash(text, n):
    h = hashlib.sha512(text.encode('UTF-8'))
    return int(h.hexdigest(), 16) % n

overall_tr_dataset = IODatasetFromFile(opt_training_data)
overall_te_dataset = IODatasetFromFile(opt_test_data)

if opt_seed_for_data_split:
    random.seed(opt_seed_for_data_split)
else:
    print('Using system\'s random initialisation')


correct = 0
total   = 0
fold    = 0
token_column = None
group2data = {}

for tr_dataset, te_dataset in overall_tr_dataset.get_folds(opt_folds):
    print('Fold with %d training items and %d test items' %(
        len(tr_dataset), len(te_dataset)
    ))
    # train model for this fold
    features, targets = tr_dataset.get_sklearn_data(context = opt_context)
    if tagger_version in (1,2):
        model = DecisionTreeClassifier(
            max_depth = max_depth,
            min_samples_leaf  = opt_leaf_size,
            random_state = 101,
        )
    elif tagger_version in (3,4):
        model = RandomForestClassifier(
            n_estimators = 100,
            min_samples_leaf = opt_leaf_size,
            random_state = 101,
        )
    model.fit(features, targets)
    if opt_viz_tree:
        feature_names = '\t'.join(tr_dataset.feature_names)
        feature_names = feature_names.replace('revr', 'reverse_rank')
        feature_names = feature_names.replace('relr', 'relative_rank')
        feature_names = feature_names.replace('\trv_idx', '\tt_reverse_idx')
        feature_names = feature_names.replace('\trelpos', '\tt_relative_pos')
        feature_names = feature_names.split('\t')
        dtreeviz(model, features, targets,
                 target_name='SE-IO-tag',
                 feature_names = feature_names,
                 class_names = labels,
       ).save('dt-%d-%d-%d-%d.svg' %(max_depth, opt_leaf_size, opt_folds, fold))
    te_features, te_gold_labels = te_dataset.get_sklearn_data(context = opt_context)
    predictions = model.predict(te_features)
    for i in range(len(te_gold_labels)):
        if te_gold_labels[i] == predictions[i]:
            correct += 1
    total += len(te_gold_labels)
    if opt_viz_tree:
        break
    # store predictions by sentence ID
    id_column = tr_dataset.header.index('item_ID')
    domain_column = tr_dataset.header.index('domain')
    if token_column is None:
        token_column = tr_dataset.header.index('token')
        rv_idx_column = tr_dataset.header.index('rv_idx')
    else:
        assert token_column == tr_dataset.header.index('token')
        assert rv_idx_column == tr_dataset.header.index('rv_idx')
    # (1) training data: use predictions for the test fold
    for index, item in enumerate(te_dataset):
        item_id = item[id_column].split(':')
        review_id = item_id[0]
        sent_idx  = int(item_id[1])
        domain = domains[int(item[domain_column])]
        group_key = ('tr', domain, review_id, sent_idx)
        if group_key not in group2data:
            group2data[group_key] = []
        group2data[group_key].append((item, predictions[index]))
    # (2) overall test data: assign fold to use by hash of sentence_id
    overall_te_features, _ = overall_te_dataset.get_sklearn_data(context = opt_context)
    predictions = model.predict(overall_te_features)
    for index, item in enumerate(overall_te_dataset):
        item_id = item[id_column].split(':')
        review_id = item_id[0]
        sent_idx  = int(item_id[1])
        domain = domains[int(item[domain_column])]
        sent_id = '%s:%s:%d' %(domain, review_id, sent_idx)
        if mhash(sent_id, opt_folds) != fold:
            continue
        group_key = ('te', domain, review_id, sent_idx)
        if group_key not in group2data:
            group2data[group_key] = []
        group2data[group_key].append((item, predictions[index]))
    fold += 1

if not opt_viz_tree:
    print('Cross-validation accuracy: %.2f%%' %(100.0* correct / float(total)))
    # write aio files
    assert token_column is not None
    sorted_group_keys = sorted(list(group2data.keys()))
    for domain in domains:
        for set_type, long_set_name in [
            ('tr', 'train'),
            ('te', 'test'),
        ]:
            f = open(
                os.path.join(opt_workdir, '%s-%s-TG%d.aio' %(long_set_name, domain, tagger_version)),
                'wt'
            )
            for group_key in sorted_group_keys:
                if group_key[0] != set_type \
                or group_key[1] != domain:
                    continue
                for item, prediction in group2data[group_key]:
                    token = item[token_column]
                    rv_idx = item[rv_idx_column]
                    label = labels[prediction]
                    f.write('%s\t%s\n' %(token, label))
                    if rv_idx == '0':  # last token of sentence
                        f.write('\n')
            f.close()


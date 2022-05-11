#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import numpy as np
import os
import random
import sys
import time
import torch

def usage():
    print('Usage: $0 [options] <SEED> <COMMAND> <MASK>')
    # TODO: print more details how to use this script

opt_saliencies_from = []
opt_save_model_as   = 'best-model-weights-only.ckpt'
opt_load_model_from = 'best-model-weights-only.ckpt'
lmf_specified = False
aio_prefix = None
seed_for_trdev_split = None  # None = use global seed
opt_lr1 = 10 / 1000000.0
opt_lr2 = 30 / 1000000.0
opt_frozen_epochs = 0
opt_vbatchsize = 64
opt_epochs = 10
while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--local-aio':
        aio_prefix = 'local-aio/'
    elif option == '--aio-prefix':
        aio_prefix = sys.argv[1]
        del sys.argv[1]
    elif option in ('--lr1', '--learning-rate-1'):
        opt_lr1 = float(sys.argv[1]) / 1000000.0
        del sys.argv[1]
    elif option in ('--lr2', '--learning-rate-2'):
        opt_lr2 = float(sys.argv[1]) / 1000000.0
        del sys.argv[1]
    elif option in ('--fre', '--frozen-epochs'):
        opt_frozen_epochs = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--vbs', '--virt-batch-size'):
        opt_vbatchsize = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--epochs'):
        opt_epochs = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--save-as', '--save-model', '--save-model-as'):
        opt_save_model_as = sys.argv[1]
        if not lmf_specified:
            opt_load_model_from = sys.argv[1]   # also evaluate this model unless --load-from specified
        del sys.argv[1]
    elif option in ('--load-from', '--load-model', '--load-model-from'):
        opt_load_model_from = sys.argv[1]
        lmf_specified = True
        del sys.argv[1]
    elif option == '--saliencies-from':
        opt_saliencies_from.append(sys.argv[1])
        del sys.argv[1]
    elif option in ('--trdev-seed', '--seed-for-tr-dev-split'):
        seed_for_trdev_split = int(sys.argv[1])
        del sys.argv[1]
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

seed = int(sys.argv[1])
assert 0 < seed < 2**31
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

command = sys.argv[2]
if command == 'train':
    skip_training = False
    skip_evaluation = False
    skip_saliency = True
elif command == 'eval':
    skip_training = True
    skip_evaluation = False
    skip_saliency = True
elif command == 'saliency':
    skip_training = True
    skip_evaluation = True
    skip_saliency = False
else:
    raise ValueError('unknown command %s' %command)

training_task = sys.argv[3]
if training_task == 'Full':     # sequence B is the review sentence as is
    training_masks = [None]
elif training_task == 'SE':     # sequence B is only the sentiment expression, all other words are masked
    training_masks = ['O']
elif training_task == 'Other':  # sequence B is all but the SE (the SE words are masked)
    training_masks = ['I']
elif training_task == 'All':    # concatenation of above training sets
    training_masks = [None, 'O', 'I']
else:
    raise ValueError('unknown training task %s' %training_task)

exclude_function_words = True    # only affects evaluation measures and confusion matrices
get_training_saliencies = True   # also print saliencies for training data in addition to dev/test

# 1.1 BERT Configuration

model_size          = 'base'  # choose between 'tiny', 'base' and 'large'
max_sequence_length = 256
batch_size          = 8       # 10 should work on a 12 GB card if not also used for graphics / GUI
virtual_batch_size  = opt_vbatchsize

# TODO: what virtual batch size should we use
# https://arxiv.org/abs/1904.00962 are cited to say a large batch size (32k) is better
# but they themselves warn that "naively" doing so can degrade performance

max_epochs = opt_epochs
limit_train_batches = 1.0  # fraction of training data to use, e.g. 0.05 during debugging

hparams = {
    "encoder_learning_rate": opt_lr1,  # Encoder specific learning rate
    "learning_rate":         opt_lr2,  # Classification head learning rate
    "nr_frozen_epochs":      opt_frozen_epochs,      # Number of epochs we want to keep the encoder model frozen
    "loader_workers":        4,      # How many subprocesses to use for data loading.
                                     # (0 means that the data will be loaded in the main process)
    "batch_size":            batch_size,
    "gpus":                  1,
}

# compensate for small batch size with batch accumulation if needed
accumulate_grad_batches = 1
while batch_size * accumulate_grad_batches < virtual_batch_size:
    # accumulated batch size too small
    # --> accumulate more batches
    accumulate_grad_batches += 1

print('Batch size:', batch_size)
if accumulate_grad_batches > 1:
    print('Accumulating gradients of %d batches' %accumulate_grad_batches)

size2name = {
    'tiny':  'distilbert-base-uncased',
    'base':  'bert-base-uncased',
    'large': 'bert-large-uncased',
}

model_name = size2name[model_size]

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

tokeniser = AutoTokenizer.from_pretrained(model_name)

# 1.2 Dataset Configuration

domains = ['laptop', 'restaurant']

train_dev_split = (95, 5)

data_prefix = 'data/'
if aio_prefix is None:
    aio_prefix = data_prefix

filenames = {
    'laptop':     ((data_prefix + 'ABSA16_Laptops_Train_SB1_v2.xml',
                    aio_prefix  + 'train.laptop.aio'
                   ),
                   (data_prefix + 'EN_LAPT_SB1_TEST_.xml.gold',
                    aio_prefix  + 'test.laptop.aio'
                   )
                  ),

    'restaurant': ((data_prefix + 'ABSA16_Restaurants_Train_SB1_v2.xml',
                    aio_prefix  + 'train.restaurant.aio'
                   ),
                   (data_prefix + 'EN_REST_SB1_TEST.xml.gold',
                    aio_prefix  + 'test.restaurant.aio'
                   )
                  ),
}

for domain in domains:
    for part in (0,1):
        for filename in filenames[domain][part]:
            print('Using', filename)


# 1.3 Question Templates

put_question_first = True  # whether to put question into seq A or B

#use_questions_with_labeltype = ['polarity', 'yesno']
use_questions_with_labeltype = ['polarity']
#use_questions_with_labeltype = ['yesno']

use_question_with_description = 'Sun et al. QA-M'    # set to None to de-activate this filter
#use_question_with_description = None

templates = [

    # Hoang et al. (2019)
    {   'question': '%(entity_type)s, %(attribute_label)s',
        'label':    '%(polarity)s',
        'description': 'Hoang et al.',
    },

    # Sun et al. (2019) format 1
    {   'question': '%(entity_type)s - %(attribute_label)s',
        'label':    '%(polarity)s',
        'description': 'Sun et al. NLI-M',
    },

    # Sun et al. (2019) format 2 "NLI-M"
    {    'question': 'What do you think of the %(attribute_label)s of %(entity_type)s?',
         'label':    '%(polarity)s',
         'description': 'Sun et al. QA-M',
    },

    # Sun et al. (2019) format 3 "QA-B"
    {    'question': 'The polarity of the aspect %(attribute_label)s of %(entity_type)s is %(candidate_polarity)s.',
         'label':    '%(yesno)s',
         'description': 'Sun et al. QA-B',
    },

    # Sun et al. (2019) format 4
    {    'question': '%(entity_type)s - %(attribute_label)s - %(candidate_polarity)s',
         'label':    '%(yesno)s',
         'description': 'Sun et al. NLI-B',
    },
]

templates += [

    # Variant 1
    {    'question': 'In terms of %(attribute_label)s, what do you think of %(entity_type)s?',
         'label':    '%(polarity)s',
         'description': 'Variant 1, QA-M',
    },

    # Variant 2
    {    'question': 'What polarity has the sentiment towards the %(attribute_label)s of %(entity_type)s in the following rewview?',
         'label':    '%(polarity)s',
         'description': 'Variant 2, QA-M',
    },

    # Variant 3
    {    'question': 'Do you agree that the sentiment towards the aspect %(attribute_label)s of %(entity_type)s in the following review is %(candidate_polarity)s?',
         'label':    '%(yesno)s',
         'description': 'Variant 3, QA-B',
    },

    # Variant 4
    {    'question': 'Is there %(candidate_polarity)s sentiment towards the %(attribute_label)s aspect of %(entity_type)s?',
         'label':    '%(yesno)s',
         'description': 'Variant 4, QA-B',
    },

]

# TODO: add variants with entity type and attribute label not in ALLCAPS and
#       with _ replaced with whitespace (requires additional code)

# remove templates with wrong question type

filtered_templates = []
for template in templates:
    use_template = False
    for labeltype in use_questions_with_labeltype:
        if labeltype in template['label']:
            use_template = True
            break
    if use_question_with_description:
        if use_question_with_description != template['description']:
            use_template = False
    if use_template:
        filtered_templates.append(template)
templates = filtered_templates
print(len(templates), 'template(s) selected')


# If there are multiple domains it probably helps to include the domain in the question
# TODO: empirically test this idea

if len(domains) > 1:
    question_prefix = '%(domain)s: '
    for template in templates:
        template['question'] = question_prefix + template['question']

assert len(templates) > 0


# 2.1 Get Data Instances from XML and AIO Files

def get_annotation(aio_filename):
    f = open(aio_filename, 'r')
    tokens = []
    sea = []
    line_no = 0
    while True:
        line = f.readline()
        line_no += 1
        if line.isspace() or not line:
            if tokens:
                yield (tokens, sea)
                tokens = []
                sea = []
            if not line:
                break
        else:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError('Unexpected AIO line %d: %r' %(line_no, line))
            tokens.append(fields[0])
            sea.append(fields[1])
    f.close()

def check_alignment(text, tokens):
    text1 = text.replace(' ', '')
    text1 = text1.replace('\xa0', '')   # also remove non-breakable space
    text1 = text1.replace('Cannot ', 'cannot')  # match lowercase of tokeniser
    text2 = ''.join(tokens)
    text2 = text2.replace('-EMTCN01-', ':)')  # restore emoticon in SEA
    if text1 != text2:
        print('Mismatch %r - %r' %(text, tokens))
    #else:
    #    print('Match %r - %r' %(text, tokens))

def absa_sort_key(sentence_xml_element):
    sent_id = sentence_xml_element.get('id')
    fields = sent_id.split(':')
    assert len(fields) == 2
    return (fields[0], int(fields[1]))

# mostly implemented from scratch, some inspiration from
# https://opengogs.adaptcentre.ie/rszk/sea/src/master/lib/semeval_absa.py

from xml.etree import ElementTree

def get_dataset(
    xml_filename, aio_filename, domain,
    observed_entity_types, observed_attribute_labels,
    observed_polarities,   observed_targets
):
    xmltree = ElementTree.parse(xml_filename)
    xmlroot = xmltree.getroot()
    annotation = get_annotation(aio_filename)
    dataset = []
    for sentence in sorted(
        xmlroot.iter('sentence'), key = absa_sort_key
    ):
        sent_id = sentence.get('id')
        #print('sent_id', sent_id)
        # get content inside the first <text>...</text> sub-element
        text = sentence.findtext('text').strip()
        op_index = 0
        for opinion in sentence.iter('Opinion'):
            op_id = '%s:%d' %(sent_id, op_index)
            tokens, sea = annotation.__next__()
            check_alignment(text, tokens)
            opin_cat = opinion.get('category')
            #print('opin_cat', opin_cat)
            entity_type, attribute_label = opin_cat.split('#')
            polarity = opinion.get('polarity')
            target = opinion.get('target')
            try:
                span = (int(opinion.get('from')), int(opinion.get('to')))
            except TypeError:
                # at least one of 'from' or 'to' is missing
                span = (0, 0)
            if target == 'NULL':
                target = None
            # add to dataset
            dataset.append((
                domain,
                op_id, text, tokens, sea,
                entity_type, attribute_label,
                target, span,
                polarity
            ))
            # update vocabularies
            observed_entity_types.add(entity_type)
            observed_attribute_labels.add(attribute_label)
            observed_polarities.add(polarity)
            if target:
                observed_targets.add(target)
            op_index += 1
        #print()
    return dataset

# get training data

tr_observed_entity_types = set()
tr_observed_attribute_labels = set()
tr_observed_polarities = set()
tr_observed_targets = set()

tr_dataset = []
for domain in domains:
    xml_filename = filenames[domain][0][0]
    aio_filename = filenames[domain][0][1]
    tr_dataset += get_dataset(
        xml_filename, aio_filename, domain,
        tr_observed_entity_types, tr_observed_attribute_labels,
        tr_observed_polarities,   tr_observed_targets
    )

print('dataset size:', len(tr_dataset))
print('\nobserved entity types:',     sorted(tr_observed_entity_types))
print('\nobserved attribute labels:', sorted(tr_observed_attribute_labels))
print('\nobserved polarities:',       sorted(tr_observed_polarities))
print('\nnumber of unique targets:',  len(tr_observed_targets))

# get test data

te_observed_entity_types = set()
te_observed_attribute_labels = set()
te_observed_polarities = set()
te_observed_targets = set()

te_dataset = []
for domain in domains:
    xml_filename = filenames[domain][1][0]
    aio_filename = filenames[domain][1][1]
    te_dataset += get_dataset(
        xml_filename, aio_filename, domain,
        te_observed_entity_types, te_observed_attribute_labels,
        te_observed_polarities,   te_observed_targets
    )

print('dataset size:', len(te_dataset))
print('\nnew observed entity types:',     sorted(te_observed_entity_types - tr_observed_entity_types))
print('\nnew observed attribute labels:', sorted(te_observed_attribute_labels - tr_observed_attribute_labels))
print('\nnew observed polarities:',       sorted(te_observed_polarities - tr_observed_polarities))
print('\nnumber of unique targets:',  len(te_observed_targets))


# 2.2 Training-Dev Split

import random

if seed_for_trdev_split is None:
    seed_for_trdev_split = seed
rnd = random.Random(seed_for_trdev_split)

# find out which sentences are used more than once

sent2count = {}
for item in tr_dataset:
    domain   = item[0]
    op_id    = item[1]
    sent_id  = op_id[:op_id.rfind(':')]
    key = (domain, sent_id)
    if not key in sent2count:
        sent2count[key] = 0
    sent2count[key] += 1

# find instances for each (domain, polarity) pair,
# keeping instances that share a sentence separate

group2indices = {}
for index, item in enumerate(tr_dataset):
    domain   = item[0]
    op_id    = item[1]
    polarity = item[-1]
    sent_id  = op_id[:op_id.rfind(':')]
    if sent2count[(domain, sent_id)] == 1:
        group = (domain, polarity)
    else:
        group = (domain, 'special')
    if not group in group2indices:
        group2indices[group] = []
    group2indices[group].append(index)

# create stratified sample

rel_train_size, rel_dev_size = train_dev_split  # configured in section 1.2
rel_total = rel_train_size + rel_dev_size

tr_indices = []
dev_indices = []

for group in group2indices:
    indices = group2indices[group]
    n = len(indices)
    select = (n * rel_train_size) // rel_total
    if group[1] != 'special':
        rnd.shuffle(indices)
        tr_indices += indices[:select]
        dev_indices += indices[select:]
        added_to_tr = select
        added_to_dev = n - select
    else:
        # randomise via hash of sent_id so that items with the
        # same sentence stay together
        hash2items = {}
        for index in indices:
            item = tr_dataset[index]
            op_id   = item[1]
            sent_id = op_id[:op_id.rfind(':')]
            key = '%d:%s:%s' %(seed_for_trdev_split, domain, sent_id)
            key = key.encode('UTF-8')
            key = hashlib.sha256(key).digest()
            if key not in hash2items:
                hash2items[key] = []
            hash2items[key].append(index)
        added_to_tr = 0
        added_to_dev = 0
        for key in sorted(list(hash2items.keys())):
            if added_to_tr < select:
                tr_indices += hash2items[key]
                added_to_tr += len(hash2items[key])
            else:
                dev_indices += hash2items[key]
                added_to_dev += len(hash2items[key])
        hash2items = None  # free memory
    print('%r: split %d (%.1f%%) to %d (%.1f%%)' %(
        group, added_to_tr, 100.0*added_to_tr/float(n),
        added_to_dev, 100.0*added_to_dev/float(n),
    ))

tr_indices.sort()
dev_indices.sort()

def get_subset(dataset, indices):  # TODO: this probably can be replaced with [] slicing
    retval = []
    for index in indices:
        retval.append(dataset[index])
    return retval

dev_dataset = get_subset(tr_dataset, dev_indices)
tr_dataset  = get_subset(tr_dataset, tr_indices)

def print_distribution_and_fingerprint(dataset, prefix = ''):
    n = len(dataset)
    group2count = {}
    op_ids = []
    for item in dataset:
        domain   = item[0]
        polarity = item[-1]
        key = (domain, polarity)
        if not key in group2count:
            group2count[key] = 0
        group2count[key] += 1
        op_ids.append(item[1].encode('UTF-8'))
    for domain, label in sorted(list(group2count.keys())):
        count = group2count[(domain, label)]
        print('%s%s\t%s\t%d\t%.1f%%' %(
            prefix, domain, label, count, 100.0 * count / float(n)
        ))
    print('%sdataset fingerprint: %s' %(
        prefix, hashlib.sha256(b'\n'.join(op_ids)).hexdigest()
    ))
    op_ids.sort()
    print('%ssorted bag of items: %s' %(
        prefix, hashlib.sha256(b'\n'.join(op_ids)).hexdigest()
    ))

print()
print('Training data size:', len(tr_dataset))
print_distribution_and_fingerprint(tr_dataset, '\t')
print('Development data size:', len(dev_dataset))
print_distribution_and_fingerprint(dev_dataset, '\t')


# 2.3 PyTorch DataLoader

# basic usage of pytorch and lightning from
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# and
# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

class ABSA_Dataset(Dataset):

    def __init__(
        self,
        raw_data,
        put_question_first = True,
        template_index = -1,    # -1 = pick random template
        mask = None,            # 'I' = mask SE tokens, i.e. use other tokens only,
                                # 'O' = mask other tokens, i.e. use SE tokens only
        info = None,            # additional info to keep with each instance
    ):
        self.raw_data            = raw_data
        self.put_question_first  = put_question_first
        self.template_index      = template_index
        self.mask                = mask
        self.info                = info

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        ''' get one instance of the dataset as a custom dictionary
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
            assert isinstance(idx, int)
        domain, opinion_id, text, \
            tokens, sea, \
            entity_type, attribute_label, \
            target, span, \
            polarity = self.raw_data[idx]
        question, label = self.pick_question(
            entity_type, attribute_label, domain, polarity,
        )
        tokens = self.apply_mask(tokens, sea)
        # TODO: support adding context (previous sentences) to text
        retval = {}
        if self.put_question_first:
            retval['seq_A'] = question
            retval['seq_B'] = tokens
        else:
            retval['seq_A'] = tokens
            retval['seq_B'] = question
        retval['label'] = label
        retval['domain'] = domain
        retval['mask']  = self.mask
        retval['info']  = self.info
        retval['index'] = idx
        retval['opinion_id'] = opinion_id
        retval['tokens'] = tokens
        retval['sea']   = sea
        return retval

    def pick_question(self, entity_type, attribute_label, domain, polarity):
        global templates
        global observed_polarities
        if self.template_index < 0:
            template = random.choice(templates)
        else:
            template = templates[self.template_index]
        candidate_polarity = random.choice(list(tr_observed_polarities))
        if candidate_polarity == polarity:
            yesno = 'yes'
        else:
            yesno = 'no'
        question = template['question'] %locals()
        label    = template['label']    %locals()
        # tokenise question
        # (no need to split off punctuation as BERT always
        #  splits off non-alphanumeric characters and we
        #  do not need to map back to tokens for the question)
        question = question.split()
        return (question, label)

    def apply_mask(self, tokens, sea):
        if not self.mask:
            return tokens
        retval = []
        for index, token in enumerate(tokens):
            if sea[index] == self.mask:
                retval.append('[MASK]')
            else:
                retval.append(token)
        return retval


# wrap training and dev data

tr_dataset_objects = []
for training_mask in training_masks:
    tr_dataset_objects.append(ABSA_Dataset(
        tr_dataset,
        put_question_first = put_question_first,
        template_index = -1,  # pick question at random
        mask = training_mask,
        info = 's%d' %seed
    ))
tr_dataset_object = torch.utils.data.ConcatDataset(tr_dataset_objects)
training_dataset_combined = tr_dataset_object  # alternative name

print('Training size:', len(tr_dataset_object))

# create a separate dev set for each question template

if skip_training and skip_evaluation:
    dev_masks = [None]
else:
    dev_masks = training_masks

dev_dataset_objects = []
for training_mask in dev_masks:
    for template_index in range(len(templates)):
        dev_dataset_objects.append(ABSA_Dataset(
            dev_dataset,
            put_question_first = put_question_first,
            template_index = template_index,
            mask = training_mask,    # e.g. if we train on SE data we want to
                                     # pick the model according to the SE score
            info = 's%d,q%d' %(seed, template_index)
        ))

# also provide a dev set that is the union of the above dev sets,
# e.g. to be used for model selection

dev_dataset_combined = torch.utils.data.ConcatDataset(dev_dataset_objects)

print('Devset size (using all templates):', len(dev_dataset_combined))

# 2.4 Test Data

# create a separate test set for each question template

if skip_evaluation:
    test_masks = [None]
else:
    test_masks = (None, 'O', 'I')

te_dataset_objects = []
for mask in test_masks:
    for template_index in range(len(templates)):
        te_dataset_objects.append(ABSA_Dataset(
            te_dataset,
            put_question_first = put_question_first,
            template_index = template_index,
            mask = mask,
            info = 'q%d' %template_index
        ))

# also provide a test set that is the union of the above test sets

te_dataset_combined = torch.utils.data.ConcatDataset(te_dataset_objects)

print('Test set size (using all templates):', len(te_dataset_combined))


# 2.5 Lightning Wrapper for Training, Development and Test Data

# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py

import pytorch_lightning as pl
from torchnlp.encoders import LabelEncoder

class ABSA_DataModule(pl.LightningDataModule):

    def __init__(self, classifier, data_split = None, **kwargs):
        global use_questions_with_labeltype
        super().__init__()
        self.hparams.update(classifier.hparams)
        self.classifier = classifier
        if data_split is None:      # this happens when loading a checkpoint
            data_split = (None, None, None)
        self.data_split = data_split
        self.kwargs = kwargs
        labelset = set()
        if 'polarity' in use_questions_with_labeltype:
            for label in tr_observed_polarities:
                labelset.add(label)
        if 'yesno' in use_questions_with_labeltype:
            labelset.add('yes')
            labelset.add('no')
        print('Labelset:', labelset)
        self.label_encoder = LabelEncoder(
            sorted(list(labelset)),
            reserved_labels = [],
        )

    def train_dataloader(self) -> DataLoader:
        ''' create a data loader for the training data '''
        dataset = self.data_split[0]
        return DataLoader(
            dataset     = dataset,
            sampler     = RandomSampler(dataset),
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        ''' create a data loader for the validation data '''
        return DataLoader(
            dataset     = self.data_split[1],
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        ''' create a data loader for the test data '''
        return DataLoader(
            dataset     = self.data_split[2],
            batch_size  = self.hparams.batch_size,
            collate_fn  = self.classifier.prepare_sample,
            num_workers = self.hparams.loader_workers,
        )

# 3.1 Classifier

from transformers import AutoModel
import torch.nn as nn
import logging as log
from torchnlp.utils import lengths_to_mask
from collections import OrderedDict
from torch import optim

class Classifier(pl.LightningModule):

    #def __init__(self, hparams = None, **kwargs) -> None:
    def __init__(self, hparams = None, **kwargs) -> None:
        super().__init__()
        if type(hparams) is dict:
            #print('Converting', type(hparams))
            hparams = pl.utilities.AttributeDict(hparams)
        #print('New classifier with', hparams)
        # https://discuss.pytorch.org/t/pytorch-lightning-module-cant-set-attribute-error/121125
        if hparams:
            self.hparams.update(hparams)
        self.add_missing_hparams()
        hparams = self.hparams
        self.batch_size = hparams.batch_size
        self.data = ABSA_DataModule(self, **kwargs)
        if 'tokeniser' in kwargs:
            self.tokenizer = kwargs['tokeniser']  # attribute expected by lightning
        else:
            # this happens when loading a checkpoint
            self.tokenizer = None  # TODO: this may break ability to use the model
        self.__build_model()
        self.__build_loss()
        # prepare training with frozen BERT layers so that the new
        # classifier head can first adjust to BERT before BERT
        # adjusts to the classifier in later epochs
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs
        self.record_predictions = False

    def __build_model(self) -> None:
        ''' Init BERT model, tokeniser and classification head '''
        # Q: Why not use AutoModelForSequenceClassification?
        self.bert = AutoModel.from_pretrained(
            model_name,  # was: self.hparams.encoder_model
            output_hidden_states = True
        )
        # parameters for the classification head: best values
        # depend on the task and dataset; the below values
        # have not been tuned much but work reasonable well
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 1536),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1536, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, self.data.label_encoder.vocab_size)
        )

    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        if self._frozen:
            log.info('\n== Encoder model fine-tuning ==')
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        ''' make a prediction for a single data instance '''
        if self.training:
            self.eval()
        with torch.no_grad():
            batch_inputs, _ = self.prepare_sample(
                [sample],
                prepare_target = False
            )
            model_out = self.forward(batch_inputs)
            logits = torch.Tensor.cpu(model_out["logits"]).numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in numpy.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]
        return sample

    # functionality to obtain predictions for a dataset as a
    # side effect of asking PyTorch Lightning to get evaluation
    # results for a dataset
    # (the framework does not seem to provide a function to get
    # all predictions for a dataset)

    def start_recording_predictions(self):
        self.record_predictions = True
        self.reset_recorded_predictions()

    def stop_recording_predictions(self):
        self.record_predictions = False

    def reset_recorded_predictions(self):
        self.seq2label = {}

    def forward(self, batch_input):
        tokens  = batch_input['input_ids']
        lengths = batch_input['length']
        mask = batch_input['attention_mask']
        # Run BERT model.
        word_embeddings = self.bert(tokens, mask).last_hidden_state
        sentemb = word_embeddings[:,0]  # at position of [CLS]
        logits = self.classification_head(sentemb)
        # Hack to conveniently use the model and trainer to
        # get predictions for a test set:
        if self.record_predictions:
            logits_np = torch.Tensor.cpu(logits).numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in numpy.argmax(logits_np, axis=1)
            ]
            for index, input_token_ids in enumerate(tokens):
                key = torch.Tensor.cpu(input_token_ids).numpy().tolist()
                # truncate trailing zeros
                while key and key[-1] == 0:
                    del key[-1]
                self.seq2label[tuple(key)] = predicted_labels[index]
        return {"logits": logits}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def add_missing_hparams(self):
        # fix various attribute errors when instantiating via
        # load_from_checkpoint()
        # like self.hparams.update() but do not overwrite
        # values already set
        # TODO: check docs whether there is a parameter to
        #       request this behaviour from update()
        global hparams
        for key in hparams:
            if not hasattr(self.hparams, key):
                self.hparams[key] = hparams[key]

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """ prepare a batch of instances to pass them into the model

        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        assert len(sample) <= batch_size
        assert self.tokenizer is not None
        batch_seq_A = []
        batch_seq_B = []
        for item in sample:
            batch_seq_A.append(item['seq_A'])
            batch_seq_B.append(item['seq_B'])
        # run the tokeniser
        encoded_batch = self.tokenizer(
            batch_seq_A,
            batch_seq_B,
            is_split_into_words = True,
            return_length       = True,
            padding             = 'max_length',
            # https://github.com/huggingface/transformers/issues/8691
            return_tensors      = 'pt',
        )
        if not prepare_target:
            return encoded_batch, {}  # no target labels requested
        # Prepare target:
        batch_labels = []
        for item in sample:
            batch_labels.append(item['label'])
        assert len(batch_labels) <= batch_size
        try:
            targets = {
                "labels": self.data.label_encoder.batch_encode(batch_labels)
            }
            return encoded_batch, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        ''' perform a training step with the given batch '''
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # Q: What is this about?
        # attribute no longer exists
        #if self.trainer.use_dp or self.trainer.use_ddp2:
        #    loss_val = loss_val.unsqueeze(0)
        output = OrderedDict({"loss": loss_val})
        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_or_validation_step(self, test_type, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        ''' perform a test or validation step with the given batch '''
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        # get predictions
        y_hat = model_out["logits"]
        labels_hat = torch.argmax(y_hat, dim=1)
        # get accuracy
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # attribute no longer exists
        #if self.trainer.use_dp or self.trainer.use_ddp2:
        #    loss_val = loss_val.unsqueeze(0)
        #    val_acc = val_acc.unsqueeze(0)
        output = OrderedDict({
            test_type + "_loss": loss_val,
            test_type + "_acc":  val_acc,
            'batch_size': len(batch),
        })
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step(
            'val', batch, batch_nb, *args, **kwargs
        )

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step(
            'test', batch, batch_nb, *args, **kwargs
        )

    # validation_end() is now validation_epoch_end()
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/efd272a3cac2c412dd4a7aa138feafb2c114326f/CHANGELOG.md

    def test_or_validation_epoch_end(self, test_type, outputs: list) -> None:
        ''' calculate average loss and accuracy over all batches,
            reducing the weight of the last batch according to its
            size so that all data instances have equal influence
            on the scores
        '''
        val_loss_mean = 0.0
        val_acc_mean = 0.0
        total_size = 0
        for output in outputs:
            val_loss = output[test_type + "_loss"]
            # reduce manually when using dp
            # -- attribute no longer exists
            #if self.trainer.use_dp or self.trainer.use_ddp2:
            #    val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
            # reduce manually when using dp
            val_acc = output[test_type + "_acc"]
            #if self.trainer.use_dp or self.trainer.use_ddp2:
            #    val_acc = torch.mean(val_acc)
            # We weight the batch accuracy by batch size to not give
            # higher weight to the items of a smaller, final bacth.
            batch_size = output['batch_size']
            val_acc_mean += val_acc * batch_size
            total_size += batch_size
        val_loss_mean /= len(outputs)
        val_acc_mean /= total_size
        self.log(test_type+'_loss', val_loss_mean)
        self.log(test_type+'_acc',  val_acc_mean)

    def validation_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('val', outputs)

    def test_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('test', outputs)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
                #"weight_decay": 0.01,  # TODO: try this as it is in the BERT paper
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()


# 4.1 Training

if not skip_training:
    classifier = Classifier(
        hparams = hparams,
        # parameters for ABSA_DataModule:
        data_split = (tr_dataset_object, dev_dataset_combined, te_dataset_combined),
        # additional required parameters:
        tokeniser  = tokeniser
    )
    print('Ready.')

# https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    early_stop_callback = EarlyStopping(
        monitor   = 'val_acc',
        min_delta = 0.00,
        patience  = 7,
        verbose   = False,
        mode      = 'max',
    )

    save_top_model_callback = ModelCheckpoint(
        save_top_k = 3,
        monitor    = 'val_acc',
        mode       = 'max',
        filename   = '{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}'
    )

    trainer = pl.Trainer(
        callbacks=[early_stop_callback, save_top_model_callback],
        max_epochs = max_epochs,
        min_epochs = classifier.hparams.nr_frozen_epochs + 2,
        gpus = classifier.hparams.gpus,
        accumulate_grad_batches = accumulate_grad_batches,  # 1 = no accumulation
        limit_train_batches     = limit_train_batches,
        check_val_every_n_epoch = 1,
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/6690
        logger = pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
    )


    start = time.time()
    trainer.fit(classifier, classifier.data)
    print('Training time: %.0f minutes' %((time.time()-start)/60.0))

    print('The best model is', save_top_model_callback.best_model_path)

    print('Best validation set accuracy:', save_top_model_callback.best_model_score)

    # The following automatically loads the best weights according to
    # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    print('Test results via trainer.test():')
    results = trainer.test()  # also prints results as a side effect


    # 5.1 Save Best Model outside Logs

    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html

    # after just having run test(), the best checkpoint is still loaded but that's
    # not a documented feature so to be on the safe side for future versions we
    # need to explicitly load the best checkpoint:

    best_model = Classifier.load_from_checkpoint(
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        # the hparams including hparams.batch_size appear to have been
        # saved in the checkpoint automatically
    )
    # best_model.save_checkpoint('best.ckpt') does not exist
    # --> need to wrap model into trainer to be able to save a checkpoint

    new_trainer = pl.Trainer(
        resume_from_checkpoint = trainer.checkpoint_callback.best_model_path,
        gpus = -1,  # avoid warnings (-1 = automatic selection)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/6690
        logger = pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
    )
    new_trainer.model = best_model  # @model.setter in plugins/training_type/training_type_plugin.py

    new_trainer.save_checkpoint(
        opt_save_model_as,
        True,  # save_weights_only
        # (if saved with setting the 2nd arg to True, the checkpoint   # TODO: "False"?
        # will contain absoulte paths and training parameters)
    )

    # to just save the bert model in pytorch format and without the classification head, we follow
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
    #best_model.bert.save_pretrained('best-bert-encoder.pt')

    # TODO: the above only saves the BERT encoder, not the classification head

    # Since the lightning module inherits from pytorch, we can save the full network in
    # pytorch format:
    #torch.save(best_model.state_dict(), 'best-model.pt')

    print('Ready')


# 5.2 Load and Test Model

best_model = Classifier.load_from_checkpoint(
    checkpoint_path = opt_load_model_from
)

best_model.eval()  # enter prediction mode, e.g. turn off dropout

print(best_model.data.data_split)  # confirm the data is not saved

def test_and_print(te_dataset_object):
    test_dataloader = DataLoader(
        dataset     = te_dataset_object,
        batch_size  = best_model.hparams.batch_size,
        collate_fn  = best_model.prepare_sample,
        num_workers = best_model.hparams.loader_workers,
    )
    #print('number of batches:', len(test_dataloader))
    new_trainer = pl.Trainer(
        gpus = -1,
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/6690
        logger = pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
    )
    if best_model.tokenizer is None:
        #print('setting tokeniser')
        best_model.tokenizer = tokeniser
    result = new_trainer.test(best_model, test_dataloaders = [test_dataloader])    # TODO: `trainer.test(test_dataloaders)` is deprecated in v1.4 and will be removed in v1.6. Use `trainer.test(dataloaders)` instead.
    assert len(result) == 1
    return result[0]['test_acc']

def get_subset_by_domain(dataset_object, domain):
    indices = []
    for index in range(len(dataset_object)):
        if domain == dataset_object[index]['domain']:
            indices.append(index)
    return torch.utils.data.Subset(dataset_object, indices)

summary = []
header = []
header.append('SeqB')
header.append('Q')
header.append('Overall')
if len(domains) > 1:
    for domain in sorted(list(domains)):
        header.append(domain.title())
header.append('Description')
summary.append('\t'.join(header))

mask2seqb = {
    None: 'Full',
    'O':  'SE',
    'I':  'Other',
}

for te_index, te_dataset_object in enumerate(te_dataset_objects):
    if skip_evaluation:
        break
    template_index = te_index % len(templates)
    row = []
    row.append(mask2seqb[te_dataset_object.mask])
    row.append('%d' %template_index)
    question = templates[template_index]['question']
    print('\nQuestion template %d: %r' %(te_index, question))
    score = 100.0 * test_and_print(te_dataset_object)
    row.append('%.9f' %score)
    if len(domains) > 1:
        print('\nBreakdown by domain:')
        for domain in sorted(list(domains)):
            print(domain)
            score = 100.0 * test_and_print(
                get_subset_by_domain(te_dataset_object, domain)
            )
            row.append('%.9f' %score)
    print()
    row.append(templates[template_index]['description'])
    summary.append('\t'.join(row))
if not skip_evaluation:
    print('\nSummary:')
    print('\n'.join(summary))

if skip_saliency:
    sys.exit(0)

# Saliency maps for dev and test data

# 6.1 Copy of Test Data with Prediction as Label
#
#     We create datasets annotated with the predicted label so the
#     gradient can be easily calculated for the predicted label
#     instead of the gold label.

# TODO


# 6.2 Get Input Gradient

def get_embedding_layer(model):
    return model.bert.embeddings
    #return model.bert.embeddings.word_embeddings

# EMNLP 2020 interpretatibility video at 3:09:52
# with modifications to

def get_gradients(model, instances, labels):
    embedding_gradients = []
    def grad_hook(module, grad_in, grad_out):
        embedding_gradients.append(grad_out[0])
    embedding = get_embedding_layer(model)
    handle = embedding.register_full_backward_hook(grad_hook)
    try:
        predictions = model(instances)
        # TODO: should we use "logits" directly to
        #       "avoid saturation issues"?
        loss = model.loss(predictions, labels)
        loss.backward()
    except:
        handle.remove()
        raise Exception
    handle.remove()
    return embedding_gradients

best_model.unfreeze_encoder()   # otherwise gradients will not arrive at embedding layer

# this seems to be running CPU-only by default


# next step is then at 3:12:38 to add a forward hook and calculate the saliencies

import numpy

def dot_and_normalise(gradients, forward_embeddings):
    # https://discuss.pytorch.org/t/how-to-do-elementwise-multiplication-of-two-vectors/13182
    products = gradients * forward_embeddings
    # adjusted from
    # https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py
    # GradientNorm._interpret()
    norms = torch.linalg.norm(products, axis=2)
    sums_of_norms = torch.sum(norms, 1)
    assert len(sums_of_norms.shape) == 1
    # https://discuss.pytorch.org/t/tensor-division-in-batches/18392
    broadcasted_sums = sums_of_norms.view(sums_of_norms.shape[0], 1)
    normalised_norms = norms / broadcasted_sums
    return normalised_norms # , norms, products

def get_alphas(variant):
    if variant == 'integrated':
        return numpy.linspace(0, 1, num=25)
    elif variant == 'point':
        return [1]
    elif variant == 'short line':
        return numpy.linspace(0.95, 1.05, num=15)
    else:
        raise ValueError('gradient-based saliency maps with %s not implemented' %variant)

# bug in the code presented in the tutorial:
#  * the stored forward embedding is multiplied by 0, causing the dot product
#    to be 0 as well and the normalised saliency values to be NaN (0/0);
#    we fix this by storing the embedding for alpha == 1 instead of
#    alpha == 0

def interpret(model, instances, labels, variant = 'integrated'):
    forward_embeddings = []
    gradients = []
    for alpha in get_alphas(variant):
        def forward_hook(module, inputs, outputs):
            if alpha == 1:
                forward_embeddings.append(outputs)
            else:
                outputs.mul_(alpha)
        embedding = get_embedding_layer(model)
        try:
            handle = embedding.register_forward_hook(forward_hook)
            gradients.append(get_gradients(model, instances, labels))
            handle.remove()
        except:
            handle.remove()
            raise Exception
    assert len(forward_embeddings) == 1
    forward_embeddings = forward_embeddings[0]
    # average over all the gradients obtained with different alpha values
    mean_gradients = []
    for gradient_batch in zip(*gradients):
        stack_of_gradients = torch.stack(gradient_batch)
        mean_gradients.append(torch.mean(stack_of_gradients, 0))
    assert len(mean_gradients) == 1
    mean_gradients = mean_gradients[0]
    return dot_and_normalise(
        mean_gradients,
        forward_embeddings,
    )

def get_dev_and_test_instances():
    global dev_dataset_combined
    global te_dataset_combined
    for test_type, test_set, masks in [
        ('training',  training_dataset_combined, training_masks),
        ('dev',       dev_dataset_combined,      dev_masks),
        ('test',      te_dataset_combined,       test_masks),
    ]:
        if test_type == 'training' and not get_training_saliencies:
            continue
        size = len(test_set) // len(masks)
        assert len(test_set) % len(masks) == 0
        for index, item in enumerate(test_set):
            item['test_type'] = test_type
            item['set_size_per_mask']  = size
            yield item

def get_batches_for_saliency(model):
    batch = []
    for item in get_dev_and_test_instances():
        batch.append(item)
        if len(batch) == batch_size:
            yield prepare_batch_for_salience(batch, model)
            batch = []
    if batch:
        yield prepare_batch_for_salience(batch, model)

def prepare_batch_for_salience(batch, model):
    start_t = time.time()
    # (1) get predictions
    #     (following ABSA_Classifier.predict())
    with torch.no_grad():
        finalsed_instances, _ = model.prepare_sample(
            batch, prepare_target = False,
        )
        model_out = model(finalsed_instances)
        logits = torch.Tensor.cpu(model_out["logits"]).numpy()
        predictions = [
            model.data.label_encoder.index_to_token[prediction]
            for prediction in numpy.argmax(logits, axis=1)
        ]
    # (2) update labels without changing gold label
    #     (if label is different, make a copy of the dictionary
    #     and change it only in the copy, making a new batch)
    new_batch = []
    updated = 0
    for index, item in enumerate(batch):
        item['gold'] = item['label']
        prediction = predictions[index]
        if prediction != item['label']:
            item = item.copy()
            item['label'] = prediction
            updated += 1
        new_batch.append(item)
    print('Saliency batch of %d needed %d label update(s).' %(len(batch), updated))
    print('Spent %.1f seconds on batch preparation.' %(time.time() - start_t))
    return new_batch

if best_model.tokenizer is None:
    #print('setting tokeniser')
    best_model.tokenizer = tokeniser

def get_confusion_matrix(sea, rationale, raw_tokens):
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


def get_token_index_for_subword_index(word_ids, start_seqB, end_seqB, index):
    assert start_seqB <= index < end_seqB
    word_id_q = word_ids[index]
    word_id_0 = word_ids[start_seqB]
    assert word_id_q is not None
    assert word_id_0 is not None
    return word_id_q - word_id_0

function_words = set()
with open(data_prefix + 'function-words.txt', 'rt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        function_words.add(line.split()[0])

example = """
Subword units: [CLS] laptop : what do you think of the design _ features of laptop ? [SEP] it ' s so nice to look at and the keys are easy to type with . [SEP]
Scores: [(0.021130122244358063, 16), (0.011964544653892517, 17), (0.01930064894258976, 18), (0.04194259271025658, 19), (0.0673951506614685, 20), (0.02628898434340954, 21), (0.039754629135131836, 22), (0.027722831815481186, 23), (0.028412630781531334, 24), (0.011970452032983303, 25), (0.06206967309117317, 26), (0.027111921459436417, 27), (0.08547540009021759, 28), (0.020255954936146736, 29), (0.051898688077926636, 30), (0.01672768034040928, 31), (0.016444828361272812, 32)]
"""

subwords2scores = {}
for saliency_file in opt_saliencies_from:
    with open(saliency_file, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('Subword units: [CLS] '):
                #print('Reading', line)
                subwords = line.split()[2:]
                assert len(subwords) > 0
                fields = f.readline().split(',')
                assert len(fields) % 2 == 0
                scores = []
                start = -1
                for index in range(len(fields)//2):
                    _, score = fields[2*index].split('(')
                    offset, _ = fields[2*index+1].split(')')
                    offset = int(offset)
                    #print('Index-Score-Offset-START:', index, score, offset, start)
                    if index == 0:
                        start = offset
                    else:
                        assert offset == start + index
                    scores.append((float(score), offset))
                assert len(subwords) > 3 + len(scores)
                subwords2scores[' '.join(subwords)] = scores

def print_example_rationales(rationale, raw_tokens, batch_item, sea):
    ''' print example rationales in I/O tag format '''
    length = len(rationale)
    for (iolabel, t_length) in [
        ('L20', int(0.20*len(raw_tokens)+0.5)),
        ('L25', int(0.25*len(raw_tokens)+0.5)),
        ('L33', int(0.33*len(raw_tokens)+0.5)),
        ('L40', int(0.40*len(raw_tokens)+0.5)),
        ('L50', int(0.50*len(raw_tokens)+0.5)),
        ('L67', int(0.67*len(raw_tokens)+0.5)),
        ('L75', int(0.75*len(raw_tokens)+0.5)),
    ]:
        if length == t_length:
            for t_index in range(len(raw_tokens)):
                row = []
                row.append(iolabel)
                row.append(batch_item['domain'])
                row.append(batch_item['test_type'])
                row.append(batch_item['opinion_id'])
                row.append(raw_tokens[t_index])
                row.append('I' if t_index in rationale else 'O')
                row.append(sea[t_index])  # also print SEA for comparison
                print('\t'.join(row))
            print()

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

summaries = {}

for batch in get_batches_for_saliency(best_model):
    summaries_updated_in_batch = set()
    print('\n\n== Batch ==\n')
    start_t = time.time()
    finalised_instance, labels = best_model.prepare_sample(
        sample = batch,
        prepare_target = True
    )
    print('Spent %.1f seconds on batch finalisation.' %(time.time() - start_t))
    if not subwords2scores:
        start_t = time.time()
        s = interpret(
            best_model, finalised_instance, labels,
            variant = 'short line'
        )
        print('Spent %.1f seconds on obtaining saliency scores.' %(time.time() - start_t))
    else:
        s = None
    #print('shape of saliencies:', s.shape)

    # display salience maps (plain text)

    start_t = time.time()
    n = labels['labels'].shape[0]
    print('n =', n)
    for j in range(n):
        print('\n\n=== Item ===\n')
        batch_item = batch[j]
        word_ids = finalised_instance.word_ids(batch_index = j)
        info = batch_item['info'].split(',')
        if len(info) == 2:
            seed, question = info
        elif info[0].startswith('s'):
            seed = info[0]
            question = '-'
        elif info[0].startswith('q'):
            seed = '-'
            question = info[0]
        else:
            raise ValueError('unsupported info %r for batch index %d' %(info, j))
        tokens = finalised_instance[j].tokens
        start_seqB = tokens.index('[SEP]') + 1
        end_seqB   = tokens.index('[PAD]') - 1
        tokens = tokens[:end_seqB + 1]   # remove padding
        sea = batch_item['sea']
        print('\t'.join('seed question set index domain mask gold pred tokens subwords SEA-I SEA-O SEA-percentage'.split()))
        print('\t'.join([
            seed, question,
            batch_item['test_type'],
            '%d' %batch_item['index'],
            batch_item['domain'],
            'None' if batch_item['mask'] is None else batch_item['mask'],
            batch_item['gold'],
            batch_item['label'],
            '%d' %(len(sea)),
            '%d' %(end_seqB - start_seqB),
            '%d' %(sea.count('I')),
            '%d' %(sea.count('O')),
            '%14.9f' %(100.0*sea.count('I')/float(len(sea))),
        ]))
        print()
        print('Subword units:', ' '.join(tokens))
        if not subwords2scores:
            scores = []
            total_pad = 0.0
            total_other = 0.0
            for i in range(s.shape[1]):
                score = s[j][i].item()
                if i < start_seqB:
                    total_other += score
                elif i < end_seqB:
                    scores.append((score, i))
                else:
                    total_pad += score
            score_pad_1 = s[j][start_seqB-1]
            score_pad_2 = s[j][end_seqB]
        else:
            scores = subwords2scores[' '.join(tokens)]
            total_pad   = -1.0   # TODO: also read these from saliency files
            total_other = -1.0
            score_pad_1 = -1.0
            score_pad_2 = -1.0
        print('Scores:', scores)
        print()
        scores_in_seqB = list(map(lambda x: x[0], scores))
        scores.sort()
        total_seqB = sum(map(lambda x: x[0], scores))  # add in order of magnitude for best numerical precision
        scores.reverse()
        top_i = list(map(lambda x: x[1], scores))
        for _, i in scores:
            top_i.append(i)
        raw_tokens = batch_item['tokens']
        for i in range(start_seqB, end_seqB):
            score = scores_in_seqB[i-start_seqB]
            top = top_i.index(i)
            sea_index = get_token_index_for_subword_index(word_ids, start_seqB, end_seqB, i)
            raw_token = raw_tokens[sea_index]
            top = '%4d' %(1+top)
            print('%s\t%14.9f\t%14.9f\t%r\t%s\t%s\t%s' %(
                top, (100.0*score), (100.0*score/total_seqB),
                sea[sea_index],
                'f' if raw_token.lower() in function_words else '  --->',
                tokens[i],
                raw_token
            ))
        print('total\t%14.9f\n\noutside seq B\t%14.9f\nbefore\t%14.9f\nafter\t%14.9f\nfirst SEP\t%14.9f\nsecond SEP\t%14.9f' %(
            100.0*total_seqB, 100.0*(1.0-total_seqB),
            100.0*total_other, 100.0*total_pad,
            100.0*score_pad_1,
            100.0*score_pad_2,
        ))
        print()
        # get evaluation metrics for every possible rationale length
        rationale = set()
        length2confusions = {}
        length2confusions[0] = get_confusion_matrix(sea, rationale, raw_tokens)
        print_example_rationales(rationale, raw_tokens, batch_item, sea)
        best_lengths = []
        best_lengths.append(0)
        best_fscore = get_fscore(length2confusions[0])
        for score, index in scores:
            # add token to rationale
            rationale.add(get_token_index_for_subword_index(word_ids, start_seqB, end_seqB, index))
            length = len(rationale)
            if length not in length2confusions:
                # found a new rationale
                # --> get confusion matrix for this rationale
                length2confusions[length] = get_confusion_matrix(sea, rationale, raw_tokens)
                # print example tables for selected lengths
                print_example_rationales(rationale, raw_tokens, batch_item, sea)
                # track lengths with best f-score
                f_score = get_fscore(length2confusions[length])
                if f_score > best_fscore:
                    best_lengths = []
                    best_lengths.append(length)
                    best_fscore = f_score
                elif f_score == best_fscore:
                    best_lengths.append(length)
            assert length + 1 == len(length2confusions)
        assert len(rationale) == len(sea)  # last rationale should cover all tokens
        # length oracle
        best_length = best_lengths[0]
        print('Best f-score %.9f with lengths %r, shortest optimal length %d\n' %(
            best_fscore, best_lengths, best_length
        ))
        # prepare storing cumulative stats for evaluation scores for full data sets
        thresholds_and_summary_keys = [
            (1001, (seed, question, batch_item['test_type'], batch_item['mask'])),
            (1,    ('length oracle', seed, question, batch_item['test_type'], batch_item['mask'])),
        ]
        for n_thresholds, summary_key in thresholds_and_summary_keys:
            if summary_key in summaries:
                continue
            summaries[summary_key] = {}
            summary = summaries[summary_key]
            for threshold in range(n_thresholds):
                d = []
                for _ in range(4):
                    d.append(0)
                for _ in range(12):
                    d.append(0.0)
                d.append(0)
                d.append(0)
                summary[threshold] = d
        summary_key = thresholds_and_summary_keys[0][1]
        summary = summaries[summary_key]
        length_oracle_summary_key = thresholds_and_summary_keys[1][1]
        length_oracle_summary = summaries[length_oracle_summary_key]
        data = []
        # print and collect stats for every possible rationale length
        print('\t'.join("""RationaleLength Percentage True-Negatives False-Positives False-Negatives
        True-Positives Precision Recall F-Score Accuracy""".split()))
        for length, length2 in enumerate(sorted(list(length2confusions.keys()))):
            assert length == length2
            row = []
            row.append('%4d' %length)
            row.append('%14.9f' %(100.0*length/float(len(sea))))
            tn, fp, fn, tp = length2confusions[length]
            row.append('%d' %tn)
            row.append('%d' %fp)
            row.append('%d' %fn)
            row.append('%d' %tp)
            # derived metrics: precision, recall, f-score and accuracy
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
            try:
                a = (tp+tn)/float(tn+fp+fn+tp)
            except ZeroDivisionError:
                a = 1.0
            row.append('%14.9f' %(100.0*p))
            row.append('%14.9f' %(100.0*r))
            row.append('%14.9f' %(100.0*f))
            row.append('%14.9f' %(100.0*a))
            print('\t'.join(row))
            row = (tn, fp, fn, tp, p, r, f, a)
            data.append(row)
            if length == best_length:
                d = length_oracle_summary[0]
                for k in range(8):
                    d[k] += row[k]
                for k in range(4):
                    d[8+k]  += (row[k] / float(len(sea)))
                    d[12+k] += (row[4+k] * float(len(sea)))
                d[16] += len(sea)
                d[17] += 1
                if 'set_size_per_mask' not in length_oracle_summary:
                    length_oracle_summary['set_size_per_mask'] = batch_item['set_size_per_mask']
                summaries_updated_in_batch.add(length_oracle_summary_key)
        # update summary stats for thresholds in steps of 0.001
        # (using integer operations to avoid numerical issues)
        for threshold in range(1001):
            d = summary[threshold]
            r_length = (len(sea) * threshold + 500) // 1000
            row = data[r_length]
            for k in range(8):
                d[k] += row[k]
            for k in range(4):
                d[8+k]  += (row[k] / float(len(sea)))
                d[12+k] += (row[4+k] * float(len(sea)))
            d[16] += len(sea)
            d[17] += 1
        if 'set_size_per_mask' not in summary:
            summary['set_size_per_mask'] = batch_item['set_size_per_mask']
        summaries_updated_in_batch.add(summary_key)
    print()
    for summary_key in summaries_updated_in_batch:
        summary = summaries[summary_key]
        if summary[0][17] < summary['set_size_per_mask']:
            print('\n\n=== Updated summary for %r ==\n' %(summary_key,))
        else:
            print('\n\n=== Final summary for %r ==\n' %(summary_key,))
        print('For %d of %d test items' %(summary[0][17], summary['set_size_per_mask']))
        header = """From To tn fp fn tp Pr Re F Acc
        Avg-Pr Avg-Re Avg-F Avg-Acc
        IW-tn IW-fp IW-fn IW-tp IW-Pr IW-Re IW-F IW-Acc
        """.split()
        print()
        print('\t'.join(header))
        last_d = None
        rows_without_header = 0
        if len(summary.keys()) > 500:
            n_thresholds = 1001
        else:
            n_thresholds = 1
        for threshold in range(1+n_thresholds):
            if (threshold % 40 == 0) and 0 < threshold < 1000 and rows_without_header > 10:
                print('#'+('\t'.join(header)))
                rows_without_header = 0
            if threshold == n_thresholds:
                d = None
            else:
                d = summary[threshold]
            if d != last_d:
                if last_d:
                    assert threshold_min is not None
                    row = []
                    row.append('%5.1f' % (threshold_min/10.0)) # from
                    row.append('%5.1f' % ((threshold-1)/10.0)) # to
                    for k in range(4):
                        row.append('%d' %(last_d[k]))  # totals of tn, fp, etc.
                    # derived metrics: precision, recall, f-score and accuracy
                    tn, fp, fn, tp = last_d[:4]
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
                    try:
                        a = (tp+tn)/float(tn+fp+fn+tp)
                    except ZeroDivisionError:
                        a = 1.0
                    row.append('%14.9f' %(100.0*p))
                    row.append('%14.9f' %(100.0*r))
                    row.append('%14.9f' %(100.0*f))
                    row.append('%14.9f' %(100.0*a))
                    # average P, R, F, A
                    row.append('%14.9f' %(100.0*last_d[4]/float(last_d[17])))
                    row.append('%14.9f' %(100.0*last_d[5]/float(last_d[17])))
                    row.append('%14.9f' %(100.0*last_d[6]/float(last_d[17])))
                    row.append('%14.9f' %(100.0*last_d[7]/float(last_d[17])))
                    # inversely weighted stats
                    for k in range(8,12):
                        row.append('%.6f' %(last_d[k]))  # totals of inversly weighted tn, fp, etc.
                    tn, fp, fn, tp = last_d[8:12]
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
                    try:
                        a = (tp+tn)/float(tn+fp+fn+tp)
                    except ZeroDivisionError:
                        a = 1.0
                    row.append('%14.9f' %(100.0*p))
                    row.append('%14.9f' %(100.0*r))
                    row.append('%14.9f' %(100.0*f))
                    row.append('%14.9f' %(100.0*a))
                    print('\t'.join(row))
                    rows_without_header += 1
                last_d = d
                if d:
                    threshold_min = threshold
                else:
                    threshold_min = None
        print('\t'.join(header))
    print()
    print('Spent %.1f seconds on printing tables.' %(time.time() - start_t))
    sys.stdout.flush()

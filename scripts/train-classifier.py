#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import numpy as np
import os
import random
import sys
import time
import torch

seed = int(sys.argv[1])
assert 0 < seed < 2**31
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

command = sys.argv[2]
if command == 'train':
    skip_training = False
    skip_evaluation = False
elif command == 'eval':
    skip_training = True
    skip_evaluation = False
elif command == 'saliency':
    skip_training = True
    skip_evaluation = True
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

# 1.1 BERT Configuration

model_size          = 'base'  # choose between 'tiny', 'base' and 'large'
max_sequence_length = 256
batch_size          = 8       # 10 should work on a 12 GB card if not also used for graphics / GUI
virtual_batch_size  = 64

# TODO: what virtual batch size should we use
# https://arxiv.org/abs/1904.00962 are cited to say a large batch size (32k) is better
# but they themselves warn that "naively" doing so can degrade performance

max_epochs = 10
limit_train_batches = 1.0     # fraction of training data to use

hparams = {
    "encoder_learning_rate": 1e-05,  # Encoder specific learning rate
    "learning_rate":         3e-05,  # Classification head learning rate
    "nr_frozen_epochs":      3,      # Number of epochs we want to keep the encoder model frozen
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
#domains = ['restaurant']

train_dev_split = (95, 5)

data_prefix = 'data/'

filenames = {
    'laptop':     (('ABSA16_Laptops_Train_SB1_v2.xml',
                    'train.laptop.aio'
                   ),
                   ('EN_LAPT_SB1_TEST_.xml.gold',
                    'test.laptop.aio'
                   )
                  ),

    'restaurant': (('ABSA16_Restaurants_Train_SB1_v2.xml',
                    'train.restaurant.aio'
                   ),
                   ('EN_REST_SB1_TEST.xml.gold',
                    'test.restaurant.aio'
                   )
                  ),
}

for domain in domains:
    for part in (0,1):
        for filename in filenames[domain][part]:
            filename = data_prefix + filename
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

def get_alignment(text, annotation):
    tokens, sea = annotation
    text1 = text.replace(' ', '')
    text1 = text1.replace('\xa0', '')   # also remove non-breakable space
    text2 = ''.join(tokens)
    text2 = text2.replace('-EMTCN01-', ':)')  # restore emoticon in SEA
    if text1 != text2:
        print('Mismatch %r - %r' %(text, tokens))
    #else:
    #    print('Match %r - %r' %(text, tokens))
    return tokens, sea

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
        for opinion in sentence.iter('Opinion'):
            tokens, sea = get_alignment(text, annotation.__next__())
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
                sent_id, text, tokens, sea,
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
        #print()
    return dataset

# get training data

tr_observed_entity_types = set()
tr_observed_attribute_labels = set()
tr_observed_polarities = set()
tr_observed_targets = set()

tr_dataset = []
for domain in domains:
    xml_filename = data_prefix + filenames[domain][0][0]
    aio_filename = data_prefix + filenames[domain][0][1]
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
    xml_filename = data_prefix + filenames[domain][1][0]
    aio_filename = data_prefix + filenames[domain][1][1]
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

# how many instances are there for each label?

group2indices = {}
for index, item in enumerate(tr_dataset):
    domain   = item[0]
    polarity = item[-1]
    group = (domain, polarity)
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
    remaining = n - select
    print('%r: split %d (%.1f%%) to %d (%.1f%%)' %(
        group, select, 100.0*select/float(n),
        remaining, 100.0*remaining/float(n),
    ))
    random.shuffle(indices)
    tr_indices += indices[:select]
    dev_indices += indices[select:]

tr_indices.sort()
dev_indices.sort()

def get_subset(dataset, indices):  # TODO: this probably can be replaced with [] slicing
    retval = []
    for index in indices:
        retval.append(dataset[index])
    return retval

dev_dataset = get_subset(tr_dataset, dev_indices)
tr_dataset  = get_subset(tr_dataset, tr_indices)

print()
print('Training data size:', len(tr_dataset))
print('Development data size:', len(dev_dataset))


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
        domain, sent_id, text, \
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

print('Training size:', len(tr_dataset_object))

# create a separate dev set for each question template

dev_dataset_objects = []
for training_mask in training_masks:
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

te_dataset_objects = []
for mask in (None, 'O', 'I'):
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
        accumulate_grad_batches = accumulate_grad_batches,
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
        "best-model-weights-only.ckpt",
        True,  # save_weights_only
        # (if saved with setting the 2nd arg to True, the checkpoint   # TODO: "False"?
        # will contain absoulte paths and training parameters)
    )

    # to just save the bert model in pytorch format and without the classification head, we follow
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
    best_model.bert.save_pretrained('best-bert-encoder.pt')

    # TODO: the above only saves the BERT encoder, not the classification head

    # Since the lightning module inherits from pytorch, we can save the full network in
    # pytorch format:
    torch.save(best_model.state_dict(), 'best-model.pt')

    print('Ready')


# 5.2 Load and Test Model

best_model = Classifier.load_from_checkpoint(
    checkpoint_path = 'best-model-weights-only.ckpt'
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
    result = new_trainer.test(best_model, test_dataloaders = [test_dataloader])
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
    for test_type, test_set in [
        ('dev',  dev_dataset_combined),
        ('test', te_dataset_combined),
    ]:
        for index, item in enumerate(test_set):
            item['test_type'] = test_type
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

for batch in get_batches_for_saliency(best_model):
    start_t = time.time()
    finalised_instance, labels = best_model.prepare_sample(
        sample = batch,
        prepare_target = True
    )
    print('Spent %.1f seconds on batch finalisation.' %(time.time() - start_t))
    start_t = time.time()
    s = interpret(
        best_model, finalised_instance, labels,
        variant = 'short line'
    )
    print('Spent %.1f seconds on obtaining saliency scores.' %(time.time() - start_t))
    #print('shape of saliencies:', s.shape)

    # display salience maps (plain text)

    n = labels['labels'].shape[0]
    print('n =', n)
    for j in range(n):
        batch_item = batch[j]
        info = batch_item['info'].split(',')
        if len(info) == 2:
            seed, question = info
        elif info[0].startswith('s'):
            seed = info[0]
            question = '-'
        elif info[0].startswith('a'):
            seed = '-'
            question = info[0]
        else:
            raise ValueError('unsupported info %r for batch index %d' %(info, j))
        print()
        print('\t'.join('seed question set index domain mask gold pred'.split()))
        print('\t'.join([
            seed, question,
            batch_item['test_type'],
            '%d' %batch_item['index'],
            batch_item['domain'],
            'None' if batch_item['mask'] is None else batch_item['mask'],
            batch_item['gold'],
            batch_item['label'],
        ]))
        tokens = finalised_instance[j].tokens
        start_seqB = tokens.index('[SEP]') + 1
        end_seqB   = tokens.index('[PAD]') - 1
        tokens = tokens[:end_seqB + 1]
        print('Subword units:', ' '.join(tokens))
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
        print('Scores:', scores)
        scores.sort()
        total_seqB = sum(map(lambda x: x[0], scores))
        scores.reverse()
        top_i = list(map(lambda x: x[1], scores))
        for _, i in scores:
            top_i.append(i)
        for i in range(start_seqB, end_seqB):
            score = s[j][i].item()
            top = top_i.index(i)
            top = '%4d' %(1+top)
            print('%s\t%9.6f\t%9.6f\t%s' %(top, (100.0*score), (100.0*score/total_seqB), tokens[i]))
        print('total\t%4.1f\t(remaining %.6f are outside seq B, %.6f before and %.6f after)' %(
            100.0*total_seqB, 100.0*(1.0-total_seqB),
            100.0*total_other, 100.0*total_pad,
        ))

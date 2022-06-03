#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
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

def usage():
    print('Usage: $0')
    # shows dataset stats when run

# 1.2 Dataset Configuration

domains = ['laptop', 'restaurant']

default_data_prefix = 'data/'

def get_filenames(aio_prefix = None):
    global default_data_prefix
    if aio_prefix is None:
        aio_prefix = default_data_prefix
    return {
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

def print_filenames(filenames):
    for domain in filenames:
        for part in (0,1):
            for filename in filenames[domain][part]:
                print('Using', filename)


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
        op_index = 0
        for opinion in sentence.iter('Opinion'):
            op_id = '%s:%d' %(sent_id, op_index)
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


def get_dataset(filenames, data_index = 0):
    observed_entity_types = set()
    observed_attribute_labels = set()
    observed_polarities = set()
    observed_targets = set()
    dataset = []
    for domain in filenames:
        xml_filename = filenames[domain][data_index][0]
        aio_filename = filenames[domain][data_index][1]
        dataset += get_dataset(
            xml_filename, aio_filename, domain,
            observed_entity_types, observed_attribute_labels,
            observed_polarities,   observed_targets
        )
    return dataset, 
           observed_entity_types, observed_attribute_labels,
           observed_polarities,   observed_targets

def main():
    seed = 101
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    aio_prefix = None
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
        else:
            print('Unknown option', option)
            usage()
            sys.exit(1)
    filenames = get_filenames(aio_prefix)
    for dataset_index, dataset_name in enumerate(['trainig', 'test']):
        print('Dataset:', dataset_name)
        dataset, _, _, _, _ = get_dataset(filnames, dataset_index)
        print('\tlength:', len(dataset))

if __name__ == "__main__":
    main()


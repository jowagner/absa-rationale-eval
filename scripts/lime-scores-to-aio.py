#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys

def usage():
    print('Usage: $0 [options]')
    # TODO: print more details how to use this script

opt_verbose = False
opt_workdir = './'
opt_prefix  = 'lime-scores'
opt_sets    = [('tr', 'train', 'training'), ('te', 'test', 'test')]
opt_domains = 'laptop restaurant'.split()
opt_r_lengths  = [25, 50, 75]
opt_wordcloud  = 50
opt_normalise  = False   # should have no effect on rationales as it does not change the ranking
opt_classes    = 'negative neutral positive'.split()

while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--verbose':
        opt_verbose = True
    elif option == '--workdir':
        opt_workdir = sys.argv[1]
        del sys.argv[1]
    elif option == '--prefix':
        opt_prefix = sys.argv[1]
        del sys.argv[1]
    elif option == '--sets':
        opt_sets = []
        for item in sys.argv[1].split(':'):
            opt_sets.append(item.split('-'))
        del sys.argv[1]
    elif option == '--domains':
        opt_domains = sys.argv[1].split(':')
        del sys.argv[1]
    elif option == '--relative-lengths':
        opt_r_lengths = []
        for item in sys.argv[1].split(':'):
            opt_r_lengths.append(int(item))
        del sys.argv[1]
    elif option == '--normalise':
        opt_normalise = True
    elif option == '--no-wordcloud':
        opt_wordcloud = None
    elif option == '--wordcloud-length':
        opt_wordcloud = int(sys.argv[1])
        del sys.argv[1]
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

def abs_score_of_predicted_class(scores, probs, pred_index):
    return abs(scores[pred_index])

def support_for_predicted_class(scores, probs, pred_index):
    score_1 = scores[pred_index]
    scores = [scores[0], scores[1], scores[2]] # deep copy and ensure it can be modified
    del scores[pred_index]  # keep the other 2 scores
    score_2plus3 = sum(scores)
    return max(0.0, score_1, -0.5 * score_2plus3)

def maximum_of_absolute_scores(scores, probs, pred_index):
    return max(
        abs(scores[0]),
        abs(scores[1]),
        abs(scores[2])
    )

for set_code, set_name, set_long_name in opt_sets:   # e.g. 'tr', 'train', 'training'
    for domain in opt_domains:
        for rationale_code, score_func in [
            ('M', abs_score_of_predicted_class),
            ('S', support_for_predicted_class),
            ('X', maximum_of_absolute_scores),
        ]:
            for threshold in opt_thresholds:
                prefix = '%(set_name)s-%(domain)s-%(rationale_code)s%02d' %locals()
                prefix_path = os.path.join(opt_workdir, prefix)
                aio_file = open(prefix_path + '.aio', 'wt')
                if opt_wordcloud == threshold:
                    wcloud_file = open(prefix_path + '-wcloud.tsv', 'wt')
                else:
                    wcloud_file = None

                score_filename = '%(opt_prefix)s-%(set_code)s.out' %locals()
                score_file = open(os.path.join(opt_workdir, score_filename), 'rt')

                while True:
                    line = score_file.readline()
                    if not line:
                        break
                    if line.isspace():
                        continue
                    if not line.startswith('== item index'):
                        raise ValueError('unexpected line in %s: %r' %(score_filename, line))
                    # read item header
                    line = score_file.readline()
                    assert line.isspace()
                    tokens = None
                    item_domain = None
                    opinion_id = None
                    sea = None
                    while True:
                        line = score_file.readline()
                        assert line
                        if line.isspace():
                            break
                        if line.startwith('tokens '):
                            tokens = line[7:].split()
                        if line.startwith('domain '):
                            item_domain = line[7:].rstrip()
                        if line.startwith('sea '):
                            sea = line[4:].split()
                    assert tokens
                    assert item_domain
                    assert opinion_id
                    assert sea
                    # read predictions and scores,
                    # skip comments and explanations
                    prediction = None
                    probs = None
                    scores = []
                    while True:
                        line = score_file.readline()
                        assert line
                        if line.startswith('#') or line.isspace():
                            continue
                        if line.startswith('Prediction'):
                            fields = line.split()
                            assert len(fields) == 5
                            prediction = fields[1]
                            probs = [
                                float(fields[2]),
                                float(fields[3]),
                                float(fields[4]),
                            ]
                        elif line.startswith('Explanation'):
                            while True:
                                line = score_file.readline()
                                assert line
                                if line.isspace():
                                    break
                                assert line.startswith('(')
                        elif line.startswith('Scores:'):
                            assert prediction
                            assert probs
                            t_index = 0
                            while True:
                                line = score_file.readline()
                                if not line or line.isspace():
                                    break
                                fields = line.split()
                                assert len(fields) >= 4
                                assert int(fields[0]) == t_index
                                scores.append((
                                    float(fields[1]),
                                    float(fields[2]),
                                    float(fields[3]),
                                ))
                        else:
                            raise ValueError('unexpected line in %s (2): %r' %(score_filename, line))
                        if scores:
                            break
                    # all information ready for this item
                    assert len(scores) == len(tokens)
                    if item_domain != domain:
                        # skip items with different domain than we are
                        # looking for in this iteration
                        continue
                    # get saliency scores from LIME scores and prediction
                    p_index = opt_classes.index(prediction)
                    saliency_scores = []
                    for index in range(len(scores)):
                        score = score_func(scores, probs, p_index)
                        tiebreaker = '%s:%s:%s:%d' %(
                            set_name, domain, opinion_id, index
                        )
                        tiebreaker = hashlib.sha256(
                            tiebreaker.encode('UTF-8')
                        ).hexdigest()
                        saliency_scores.append((score, tiebreaker, index))
                    # optionally normalise scores
                    if opt_normalise:
                        total = sum(saliency_scores)
                        new_scores = []
                        for index in range(len(scores)):
                            score, tiebreaker, index = saliency_scores[index]
                            new_scores.append((
                                score / total,
                                tiebreaker,
                                index,
                            ))
                        saliency_scores = new_scores
                    # calculate absolute length for given relative length
                    r_length = (50 + len(tokens) * rel_length) // 100
                    # get rationale indices
                    saliency_scores.sort(reverse = True)
                    indices =set(map(lambda x: x[-1], saliency_scores[:r_length]))
                    # write output
                    for t_index, token in enumerate(tokens):
                        tag = 'I' if t_index in indices else 'O'
                        aio_file.write('%s\t%s\n' %(token, tag))
                        if wcloud_file is not None:
                            row = []
                            row.append('%s%02d' %(rationale_code, rel_length))
                            row.append(domain)
                            row.append(set_long_name)
                            row.append(opinion_id)
                            row.append(token)
                            row.append(tag)
                            row.append(sea[t_index])
                            wcloud_file.write('\t'.join(row))
                    aio_file.write('\n')
                    if wcloud_file is not None:
                        wcloud_file.write('\n')

                aio_file.close()
                score_file.close()
                if wcloud_file is not None:
                    wcloud_file.close()

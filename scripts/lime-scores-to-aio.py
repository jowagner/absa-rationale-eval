#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import os
import sys

import evaluation

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
data_prefix    = 'data/'
opt_write_fscores = True

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
    elif option == '--no-fscores':
        opt_write_fscores = None
    elif option == '--no-wordcloud':
        opt_wordcloud = None
    elif option == '--wordcloud-length':
        opt_wordcloud = int(sys.argv[1])
        del sys.argv[1]
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

if opt_wordcloud is not None:
    assert opt_wordcloud in opt_r_lengths

if opt_write_fscores and not opt_r_lengths:
    opt_r_lengths.append(None)

def abs_score_of_predicted_class(scores, probs, pred_index):
    return abs(scores[pred_index])

def scaled_score_of_predicted_class(scores, probs, pred_index):
    return 0.5 + 0.5 * scores[pred_index]

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

def get_seed(workdir):
    fields = workdir.split('-')
    if len(fields) == 4 and fields[0] == 'c':
        try:
            a = int(fields[2])
            b = int(fields[3])
            if a > 0 and b > 0 and b < 4:
                return '%d' %(3*(a-1) + b)
        except:
            pass
    return workdir

if opt_write_fscores:
    seed = get_seed(opt_workdir)

function_words = evaluation.init_function_words(data_prefix)

for set_code, set_name, set_long_name in opt_sets:   # e.g. 'tr', 'train', 'training'
    for rationale_code, score_func in [
        ('M', abs_score_of_predicted_class),
        ('N', scaled_score_of_predicted_class),
        ('S', support_for_predicted_class),
        ('X', maximum_of_absolute_scores),
    ]:
        for rl_index, rel_length in enumerate(opt_r_lengths):
            if opt_write_fscores \
            and rl_index == 0    \
            and set_code == 'te':
                summaries = {}
                path = '%s(opt_workdir)s/lime-%(rationale_code)s-fscores-%(set_code)s.txt' %locals()
                summary_file = open(path, 'wt')
            else:
                summaries    = None
                summary_file = None

            cross_domain_item_count = 0

            for domain in opt_domains:
                prefix = '%(set_name)s-%(domain)s-%(rationale_code)s%(rel_length)02d' %locals()
                prefix_path = os.path.join(opt_workdir, prefix)
                if rel_length is not None:
                    aio_file = open(prefix_path + '.aio', 'wt')
                else:
                    aio_file = None
                if opt_wordcloud is not None and opt_wordcloud == rel_length:
                    wcloud_file = open(prefix_path + '-wcloud.tsv', 'wt')
                else:
                    wcloud_file = None

                score_filename = '%(opt_prefix)s-%(set_code)s.out' %locals()
                score_file = open(os.path.join(opt_workdir, score_filename), 'rt')

                item_index = 0
                while True:
                    line = score_file.readline()
                    if not line:
                        break
                    if line.isspace() or line.startswith('#'):
                        continue
                    if not line.startswith('== item index'):
                        raise ValueError('unexpected line in %s: %r' %(score_filename, line))
                    # check item index
                    assert int(line.split()[3]) == item_index
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
                        #sys.stderr.write('item %d header line %r\n' %(item_index, line))
                        if line.startswith('tokens '):
                            tokens = line[7:].split()
                        elif line.startswith('domain '):
                            item_domain = line[7:].rstrip()
                        elif line.startswith('opinion_id '):
                            opinion_id = line[11:].rstrip()
                        elif line.startswith('sea '):
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
                            if not prediction:
                                raise ValueError('missing prediction in %s/%s item %d' %(opt_workdir, score_filename, item_index))
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
                                t_index += 1
                        else:
                            raise ValueError('unexpected line in %s (2): %r' %(score_filename, line))
                        if scores:
                            break
                    # all information ready for this item
                    assert len(scores) == len(tokens)
                    if item_domain != domain \
                    and not (set_code == 'te' and summaries is not None):
                        # skip items with different domain than we are
                        # looking for in this iteration
                        # (except test items needed for fscore summaries)
                        item_index += 1
                        continue
                    # get saliency scores from LIME scores and prediction
                    p_index = opt_classes.index(prediction)
                    saliency_scores = []
                    for index in range(len(scores)):
                        score = score_func(scores[index], probs, p_index)
                        tiebreaker = '%s:%s:%s:%d' %(
                            set_name, domain, opinion_id, index
                        )
                        tiebreaker = hashlib.sha256(
                            tiebreaker.encode('UTF-8')
                        ).hexdigest()
                        saliency_scores.append((score, tiebreaker, index))
                    # optionally normalise scores
                    if opt_normalise:
                        total = sum(map(lambda x: x[0], saliency_scores))
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
                    all_indices = map(lambda x: x[-1], saliency_scores)
                    indices = set(all_indices[:r_length])
                    # write output
                    for t_index, token in enumerate(tokens):
                        tag = 'I' if t_index in indices else 'O'
                        if aio_file is not None:
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
                            #for _, score_func2 in [
                            #    ('M', abs_score_of_predicted_class),
                            #    ('N', scaled_score_of_predicted_class),
                            #    ('S', support_for_predicted_class),
                            #    ('X', maximum_of_absolute_scores),
                            #]:
                            #    row.append('%.9f' %score_func2(scores[t_index], probs, p_index))
                            #sys.stderr.write('item %d token %d wc: %r\n' %(item_index, t_index, row))
                            wcloud_file.write('\t'.join(row))
                            wcloud_file.write('\n')
                    if aio_file is not None:
                        aio_file.write('\n')
                    if wcloud_file is not None:
                        wcloud_file.write('\n')
                    if summaries is not None:
                        print('\n\n=== Item ===\n')
                        # get confusion matrices for each rationale length
                        length2confusions, best_length = get_confusion_matrices(
                            sea, all_indices, tokens,
                            excl_function_words = opt_exclude_function_words,
                            out_file = summary_file,
                        )
                        assert len(length2confusions) == len(tokens) + 1
                        # prepare storing cumulative stats for evaluation scores for full data sets
                        thresholds_and_summary_keys = [
                            (1001, (seed, set_name)),
                            (1,    ('length oracle', seed, set_name)),
                        ]
                        for n_thresholds, summary_key in thresholds_and_summary_keys:
                            if summary_key in summaries:
                                continue
                            summaries[summary_key] = evaluation.FscoreSummaryTable()
                        summary_key = thresholds_and_summary_keys[0][1]
                        summary = summaries[summary_key]
                        length_oracle_summary_key = thresholds_and_summary_keys[1][1]
                        length_oracle_summary = summaries[length_oracle_summary_key]
                        # get evaluation stats such as precision and recall
                        # for each confusion length
                        data = evaluation.get_and_print_stats(
                            length2confusions,
                            best_length = best_length,
                            length_oracle_summary      = length_oracle_summary,
                            length_oracle_summary_key  = length_oracle_summary_key,
                            summaries_updated_in_batch = None,
                            out_file = summary_file,
                        )
                        assert len(data) == len(length2confusions)
                        # add to summary table
                        summary.update(len(tokens), data)

                    item_index += 1
                    cross_domain_item_count += 1

                score_file.close()
                if aio_file is not None:
                    aio_file.close()
                if wcloud_file is not None:
                    wcloud_file.close()

            if summaries is not None:
                for summary_key in summaries:
                    summary = summaries[summary_key]
                    summary_file.write('\n\n=== Final summary for %r ==\n\n' %(summary_key,))
                    summary_file.write('For %d of %d test items\n\n' %(summary[0][17], cross_domain_item_count))
                    summary.print_stats(summary_file)

            if summary_file is not None:
                summary_file.close()


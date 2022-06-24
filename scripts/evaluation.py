#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# module initialisation

function_words = set()

# public functions

def init_function_words(data_prefix = 'data/'):
    global function_words
    with open(data_prefix + 'function-words.txt', 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            function_words.add(line.split()[0])
    return function_words

def get_confusion_matrix(sea, rationale, raw_tokens, exclude_function_words = True):
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

def get_confusion_matrices(
    sea, r_indices, tokens,
    excl_function_words = True,
    out_file = None,
    print_example_rationales = None,
    item_info = None,
    subword_info = None,
    subword_to_token_index = None,
):
    ''' get evaluation metrics for every possible rationale length
    '''
    assert len(r_indices) == len(tokens)
    assert len(r_indices) == len(sea)
    if subword_info:
        assert subword_to_token is not None
    if not subword_indices:
        subword_indices = range(len(tokens))
    rationale = set()
    length2confusions = {}
    length2confusions[0] = evaluation.get_confusion_matrix(
        sea, rationale, tokens,
        excl_function_words,
    )
    if print_example_rationales is not None:
        print_example_rationales(rationale, tokens, item_info, sea)
    best_lengths = []
    best_lengths.append(0)
    best_fscore = evaluation.get_fscore(length2confusions[0])
    for index in r_indices:
        if subword_to_token_index:
            index = subword_to_token_index(subword_info, index)
        # add token to rationale
        rationale.add(index)
        length = len(rationale)
        if length not in length2confusions:
            # found a new rationale
            # --> get confusion matrix for this rationale
            length2confusions[length] = get_confusion_matrix(
                sea, rationale, tokens,
                exclude_function_words,
            )
            # print example tables for selected lengths
            if print_example_rationales is not None:
                print_example_rationales(rationale, tokens, item_info, sea)
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
    if out_file is not None:
        out_file.write('Best f-score %.9f with lengths %r, shortest optimal length %d\n' %(
            best_fscore, best_lengths, best_length
        ))
    return length2confusions, best_length

def get_and_print_stats(
    length2confusions,
    best_length = None,
    length_oracle_summary = None,
    summaries_updated_in_batch = None,
    length_oracle_summary_key  = None,
    item_info = None,
    out_file = None,
):
    ''' print and collect stats for every possible rationale length
    '''
    data = []
    if out_file is not None:
        out_file.write('\t'.join("""RationaleLength Percentage True-Negatives False-Positives False-Negatives
    True-Positives Precision Recall F-Score Accuracy""".split()))
        out_file.write('\n')
    for length, length2 in enumerate(sorted(list(length2confusions.keys()))):
        assert length == length2
        tn, fp, fn, tp = length2confusions[length]
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
        if out_file is not None:
            row = []
            row.append('%4d' %length)
            row.append('%14.9f' %(100.0*length/float(len(sea))))
            row.append('%d' %tn)
            row.append('%d' %fp)
            row.append('%d' %fn)
            row.append('%d' %tp)
            row.append('%14.9f' %(100.0*p))
            row.append('%14.9f' %(100.0*r))
            row.append('%14.9f' %(100.0*f))
            row.append('%14.9f' %(100.0*a))
            out_file.write('\t'.join(row))
            out_file.write('\n')
        row = (tn, fp, fn, tp, p, r, f, a)
        data.append(row)
        if length_oracle_summary and length == best_length:
            length_oracle_summary.update(len(sea), row, is_row = True)
            if item_info \
            and 'set_size_per_mask' in item_info \
            and 'set_size_per_mask' not in length_oracle_summary:
                length_oracle_summary['set_size_per_mask'] = item_info['set_size_per_mask']
            if summaries_updated_in_batch is not None:
                summaries_updated_in_batch.add(length_oracle_summary_key)
    return data


class FscoreSummaryTable:

    def __init__(selfi, n_thresholds = 1001):
        self.table = {}
        self.n_thresholds = n_thresholds
        for threshold in range(n_thresholds):
            d = []
            for _ in range(4):
                d.append(0)
            for _ in range(12):
                d.append(0.0)
            d.append(0)
            d.append(0)
            self.table[threshold] = d

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        if type(key) == int:
            raise KeyError('FscoreSummaryTable rejected attempt to overwrite table row')
        self.table[key] = value

    def update(self, seq_length, data, is_row = False):
        ''' update summary stats for thresholds in steps of 0.001
            (using integer operations to avoid numerical issues)
        '''
        for threshold in range(self.n_thresholds):
            d = self[threshold]
            if is_row:
                row = data
            else:
                r_length = (seq_length * threshold + 500) // 1000
                row = data[r_length]
            for k in range(8):
                d[k] += row[k]
            for k in range(4):
                d[8+k]  += (row[k] / float(seq_length))
                d[12+k] += (row[4+k] * float(seq_length))
            d[16] += seq_length
            d[17] += 1

    def print_stats(self, out_file, repeat_header = False):
        out = out_file
        header = """From To tn fp fn tp Pr Re F Acc
        Avg-Pr Avg-Re Avg-F Avg-Acc
        IW-tn IW-fp IW-fn IW-tp IW-Pr IW-Re IW-F IW-Acc
        """.split()
        out.write('\t'.join(header))
        last_d = None
        rows_without_header = 0
        threshold_min = None
        for threshold in range(1+self.n_thresholds):
            if repeat_header \
            and (threshold % 40 == 0) \
            and 0 < threshold < 1000  \
            and rows_without_header > 10:
                print('#'+('\t'.join(header)))
                rows_without_header = 0
            if threshold == self.n_thresholds:
                d = None
            else:
                d = self[threshold]
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
                    out.write('\t'.join(row))
                    out.write('\n')
                    rows_without_header += 1
                last_d = d
                if d:
                    threshold_min = threshold
                else:
                    threshold_min = None
        if repeat_header:
            out.write('\t'.join(header))
            out.write('\n')



def main():
    import sys
    confusion_matrix = map(int, sys.argv[1:5])
    sys.stdout.write('%.4f\n' %get_fscore(confusion_matrix))

if __name__ == "__main__":
    main()


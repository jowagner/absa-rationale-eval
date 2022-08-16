#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

def get_median(values):
    n = len(values)
    if n > 2:
        values = sorted(values)
    n2 = n // 2
    if n % 2 == 0:
        # even number of items
        return (values[n2-1]+values[n2]) / 2.0
    return values[n2]

def add_middle(values):
    values = sorted(values)
    new_values = []
    for index in range(1, len(values)):
        new_values.append((values[index] + values[index-1])/2.0)
    return sorted(values + new_values)

def i_q(values, median):
    # replicate Texax Instrument behaviour
    # (described in Wikipedia)
    values = sorted(values)
    n = len(values)
    n2 = n // 2
    if n % 2 == 0:
        left = values[:n2]
        right = values[n2:]
    else:
        left = values[:n2]
        right = values[n2+1:]
    assert len(left) == len(right)
    q1 = get_median(left)
    q3 = get_median(right)
    return q1, q3

def r_q(values, median):
    # replicate R behaviour as described
    # by Krzywinski and Altman 2014
    # a.k.a. Tukey's hinge according to Wikipedia
    values = sorted(values)
    n = len(values)
    n2 = n // 2
    if n % 2 == 0:
        left = values[:n2]
        right = values[n2:]
    else:
        left = values[:n2+1]
        right = values[n2:]
    assert len(left) == len(right)
    q1 = get_median(left)
    q3 = get_median(right)
    return q1, q3

def m_q(values, median):
    # my idea for obtaining quartile values
    # (1) expand list of values, adding interpolated
    #     values twice, such that the number of values
    #     becomes 5 + 4k
    values = add_middle(add_middle(values))
    assert (len(values)-5) % 4 == 0
    k = (len(values)-5) // 4
    assert median == values[2*(k+1)]
    q1 = values[k+1]
    q3 = values[3*(k+1)]
    return q1, q3

def helper_e_q(x, n, p):
    k = int(p*(n+1))
    alpha = p*(n+1) - k
    return x[k-1] + alpha*(x[k] - x[k-1])

def e_q(values, median):
    # method 4 of https://en.wikipedia.org/wiki/Quartile
    values = sorted(values)
    n = len(values)
    q1 = helper_e_q(values, n, 0.25)
    q3 = helper_e_q(values, n, 0.75)
    return q1, q3

def helper_a_q(values, include_median = False):
    n = len(values)
    if n % 2 == 0 and include_median:
        values = values[:]
        values.append(get_median(values))
        n += 1
    if n % 2 == 0:
        return r_q(values, None)
    else:
        values = sorted(values)
        k = n // 4
        if n % 4 == 1:
            q1 = 0.25 * values[k-1] + 0.75 * values[k]
            q3 = 0.75 * values[3*k] + 0.25 * values[3*k+1]
        elif n % 4 == 3:
            q1 = 0.75 * values[k] + 0.25 * values[k+1]
            q3 = 0.25 * values[3*k+1] + 0.75 * values[3*k+2]
        else:
            raise ValueError
        return q1, q3

def a_q(values, median):
    return helper_a_q(values, False)

def b_q(values, median):
    return helper_a_q(values, True)


class BoxPlot:

    def __init__(self, scores):
        self.median = get_median(scores)
        self.q1, self.q3 = e_q(scores, self.median)
        iqr = self.q3 - self.q1
        assert iqr >= 0.0
        ci95 = 1.58 * iqr / float(len(scores)**0.5)
        self.ci95 = (self.median - ci95, self.median + ci95)
        lower_cut_off = self.q1 - 1.5 * iqr
        upper_cut_off = self.q3 + 1.5 * iqr
        self.outliers = []
        self.backers  = []
        for score in scores:
            if score < lower_cut_off \
            or score > upper_cut_off:
                self.outliers.append(score)
            else:
                self.backers.append(score)
        assert len(self.backers) > 0
        self.lower_whisker = min(self.backers)
        self.upper_whisker = max(self.backers)
        self.scores = scores

    def __getitem__(self, key):
        if type(key) == tuple:
            if key[0] in ('O', 'o', 'outlier'):
                return self.outliers[key[1]]
            elif key[0] in ('I', 'i', 'inside', 'backer'):
                return self.backers[key[1]]
            elif key[0] in ('D', 'd', 'data'):
                return self.scores[key[1]]
            else:
                raise KeyError
        elif key in ('B', 'b', 'L', 'l', 'bottom-whisker', 'lower-whisker'):
            return self.lower_whisker
        elif key in ('T', 't', 'U', 'u', 'top-whisker',    'upper-whisker'):
            return self.upper_whisker
        elif key in ('Q1', 'q1', '25th-percentile'):
            return self.q1
        elif key in ('Q3', 'q3', '75th-percentile'):
            return self.q3
        elif key in ('M', 'm', 'median'):
            return self.median
        else:
            raise KeyError('unknown key "%s" for box plot' %key)


def get_synthetic_population(size, interval = 100):
    population = []
    for _ in range(size):
        values = []
        for _ in range(5):
            values.append(random.random()**8)
        population.append(1000+int(interval*sum(values)/5.0))
    return population

    population = []
    for _ in range(size):
        values = []
        values.append(1.0)
        for _ in range(4):
            values.append(random.random())
            values.append(random.random()**2)  # introducing some skew to the left
        population.append(int(200.0*sum(values)))
    return population

def get_absa_accuracies():
    f = open('data/preliminary_test_scores.txt', 'rt')
    retval = []
    while True:
        line = f.readline()
        if not line:
            break
        retval.append(float(line.split()[0]))
    f.close()
    return retval

def get_interpolated_range(size):
    values = []
    x = 100.0
    for _ in range(size):
        values.append(x)
        x += 1.0
    return values

def add_annotation(annotation, x, text):
    if not x in annotation:
        annotation[x] = []
    annotation[x].append(text)

def annotate_values(sample, annotation):
    for s_index, x in enumerate(sorted(sample)):
        if x not in annotation:
            annotation[x] = []
        annotation[x].append('x_%d' %(s_index+1))

def print_sample_with_annotation(values, annotation):
    median = get_median(values)
    add_annotation(annotation, median, 'median')
    r_q1, r_q3 = r_q(values, median)
    add_annotation(annotation, r_q1, 'R-Q1')
    add_annotation(annotation, r_q3, 'R-Q3')
    m_q1, m_q3 = m_q(values, median)
    add_annotation(annotation, m_q1, 'M-Q1')
    add_annotation(annotation, m_q3, 'M-Q3')
    i_q1, i_q3 = i_q(values, median)
    add_annotation(annotation, i_q1, 'I-Q1')
    add_annotation(annotation, i_q3, 'I-Q3')
    a_q1, a_q3 = a_q(values, median)
    add_annotation(annotation, a_q1, 'A-Q1')
    add_annotation(annotation, a_q3, 'A-Q3')
    b_q1, b_q3 = b_q(values, median)
    add_annotation(annotation, b_q1, 'B-Q1')
    add_annotation(annotation, b_q3, 'B-Q3')
    e_q1, e_q3 = e_q(values, median)
    add_annotation(annotation, e_q1, 'E-Q1')
    add_annotation(annotation, e_q3, 'E-Q3')
    # print table
    for x in sorted(list(annotation.keys())):
        row = []
        row.append('%.2f' %x)
        if annotation[x][0][:2] == 'x_':
            row.append(annotation[x][0])
            start = 1
        else:
            row.append('')
            start = 0
        if len(annotation[x]) > start:
            row.append(' '.join(annotation[x][start:]))
        sys.stdout.write('\t'.join(row))
        sys.stdout.write('\n')
    sys.stdout.write('\n')

def main():
    verbose = False
    n = int(sys.argv[1])  # sample size
    sys.stdout.write('n = %d\n' %n)

    sys.stdout.write('\n== Evenly spaced values (integers) ==\n\n')
    sample = get_interpolated_range(n)
    annotation = {}
    annotate_values(sample, annotation)
    print_sample_with_annotation(sample, annotation)

    sys.stdout.write('\n== A skewed distribution (integers) ==\n\n')
    sample = get_synthetic_population(n)
    annotation = {}
    annotate_values(sample, annotation)
    print_sample_with_annotation(sample, annotation)

    sys.stdout.write('\n== Classifier accuracies ==\n\n')
    sample = random.choices(get_absa_accuracies(), k = n)
    annotation = {}
    annotate_values(sample, annotation)
    print_sample_with_annotation(sample, annotation)

    #sys.exit(0)
    sys.stdout.flush()

    sys.stdout.write('\n== Which method works best on average? ==\n\n')
    
    # TODO: implement methods 3 and 4 from
    #       https://en.wikipedia.org/wiki/Quartile

    # let's count which method works best on average
    repetitions = 5000
    win_r = 0
    win_m = 0
    win_i = 0
    win_a = 0
    win_b = 0
    win_e = 0
    pop_unique_min = None
    pop_unique_max = None
    for _ in range(repetitions):
        if verbose:
            print('\n=== Repetition ==\n\n')
        #population = get_synthetic_population(4005, int(sys.argv[2]))  # 5 + 4k with k = 1000
        population = get_absa_accuracies()
        unique_size = len(set(population))
        if pop_unique_min is None or unique_size < pop_unique_min:
            pop_unique_min = unique_size
        if pop_unique_max is None or unique_size > pop_unique_max:
            pop_unique_max = unique_size
        median = get_median(population)
        # find true quartile values
        t_r_q1, t_r_q3 = r_q(population, median)
        t_m_q1, t_m_q3 = m_q(population, median)
        t_i_q1, t_i_q3 = i_q(population, median)
        t_a_q1, t_a_q3 = a_q(population, median)
        t_b_q1, t_b_q3 = b_q(population, median)
        t_e_q1, t_e_q3 = e_q(population, median)
        t_q1 = sum([t_r_q1, t_m_q1, t_i_q1, t_a_q1, t_b_q1, t_e_q1]) / 6.0
        t_q3 = sum([t_r_q3, t_m_q3, t_i_q3, t_a_q3, t_b_q3, t_e_q3]) / 6.0
        if verbose:
            sys.stdout.write('     \t R  \t M  \t I  \t A  \t B  \t E  \t span\n')
            sys.stdout.write('T(Q1)\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %(
                t_r_q1, t_m_q1, t_i_q1, t_a_q1, t_b_q1, t_e_q1,
                max(t_r_q1, t_m_q1, t_i_q1, t_a_q1, t_b_q1, t_e_q1) - min(t_r_q1, t_m_q1, t_a_q1, t_a_q1, t_b_q1, t_e_q1)
            ))
            sys.stdout.write('T(Q3)\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' %(
                t_r_q3, t_m_q3, t_i_q3, t_a_q3, t_b_q3, t_e_q3,
                max(t_r_q3, t_m_q3, t_i_q3, t_a_q3, t_b_q3, t_e_q3) - min(t_r_q3, t_m_q3, t_a_q3, t_a_q3, t_b_q3, t_e_q3)
            ))
            sys.stdout.write('\n')
            sys.stdout.write('pop range: %d to %d\n' %(min(population), max(population)))
            sys.stdout.write('\n')

        sum_r_q1 = 0.0
        sum_r_q3 = 0.0
        sum_m_q1 = 0.0
        sum_m_q3 = 0.0
        sum_i_q1 = 0.0
        sum_i_q3 = 0.0
        sum_a_q1 = 0.0
        sum_a_q3 = 0.0
        sum_b_q1 = 0.0
        sum_b_q3 = 0.0
        sum_e_q1 = 0.0
        sum_e_q3 = 0.0
        n_samples = 1024
        for _ in range(n_samples):
            values = random.choices(population, k = n)
            median = get_median(values)
            r_q1, r_q3 = r_q(values, median)
            m_q1, m_q3 = m_q(values, median)
            i_q1, i_q3 = i_q(values, median)
            a_q1, a_q3 = a_q(values, median)
            b_q1, b_q3 = b_q(values, median)
            e_q1, e_q3 = e_q(values, median)
            sum_r_q1 += r_q1
            sum_r_q3 += r_q3
            sum_m_q1 += m_q1
            sum_m_q3 += m_q3
            sum_i_q1 += i_q1
            sum_i_q3 += i_q3
            sum_a_q1 += a_q1
            sum_a_q3 += a_q3
            sum_b_q1 += b_q1
            sum_b_q3 += b_q3
            sum_e_q1 += e_q1
            sum_e_q3 += e_q3
        avg_r_q1 = sum_r_q1 / float(n_samples)
        avg_r_q3 = sum_r_q3 / float(n_samples)
        avg_m_q1 = sum_m_q1 / float(n_samples)
        avg_m_q3 = sum_m_q3 / float(n_samples)
        avg_i_q1 = sum_i_q1 / float(n_samples)
        avg_i_q3 = sum_i_q3 / float(n_samples)
        avg_a_q1 = sum_a_q1 / float(n_samples)
        avg_a_q3 = sum_a_q3 / float(n_samples)
        avg_b_q1 = sum_b_q1 / float(n_samples)
        avg_b_q3 = sum_b_q3 / float(n_samples)
        avg_e_q1 = sum_e_q1 / float(n_samples)
        avg_e_q3 = sum_e_q3 / float(n_samples)
        if verbose:
            sys.stdout.write('     \t R  \t error \t M  \t error \t I  \t error \t A  \t error \t B  \t error \t E  \t error \t winner\n')
        for q, t_q, t_r_q, avg_r_q, t_m_q, avg_m_q, t_i_q, avg_i_q, t_a_q, avg_a_q, t_b_q, avg_b_q, t_e_q, avg_e_q in [
            (1, t_q1, t_r_q1, avg_r_q1, t_m_q1, avg_m_q1, t_i_q1, avg_i_q1, t_a_q1, avg_a_q1, t_b_q1, avg_b_q1, t_e_q1, avg_e_q1),
            (3, t_q3, t_r_q3, avg_r_q3, t_m_q3, avg_m_q3, t_i_q3, avg_i_q3, t_a_q3, avg_a_q3, t_b_q1, avg_b_q3, t_e_q3, avg_e_q3),
        ]:
            error_and_tag = [
                (abs(t_q - avg_r_q), 'R'),
                (abs(t_q - avg_m_q), 'M'),
                (abs(t_q - avg_i_q), 'I'),
                (abs(t_q - avg_a_q), 'A'),
                (abs(t_q - avg_b_q), 'B'),
                (abs(t_q - avg_e_q), 'E'),
            ]
            error_and_tag.sort()
            best_error, winner = error_and_tag[0]
            # tie detection
            tie_index = 1
            n_winners = 1.0
            while tie_index < len(error_and_tag):
                error = error_and_tag[tie_index][0]
                if best_error < error:
                    break
                # found another member of the tie
                winner = winner + '+' + error_and_tag[tie_index][1]
                tie_index += 1
                n_winners += 1.0
            if verbose:
                sys.stdout.write('E(Q%d)\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n' %(
                    q,
                    avg_r_q, abs(t_q - avg_r_q),
                    avg_m_q, abs(t_q - avg_m_q),
                    avg_i_q, abs(t_q - avg_i_q),
                    avg_a_q, abs(t_q - avg_a_q),
                    avg_b_q, abs(t_q - avg_b_q),
                    avg_e_q, abs(t_q - avg_e_q),
                    winner,
                ))
            if 'R' in winner:
                win_r += 1.0 #/ n_winners
            elif 'M' in winner:
                win_m += 1.0 #/ n_winners
            elif 'I' in winner:
                win_i += 1.0 #/ n_winners
            elif 'A' in winner:
                win_a += 1.0 #/ n_winners
            elif 'B' in winner:
                win_b += 1.0 #/ n_winners
            elif 'E' in winner:
                win_e += 1.0 #/ n_winners
            else:
                raise ValueError
    
        if verbose:
            sys.stdout.write('\n')
    
    sys.stdout.write('unique size range %d - %d\n' %(pop_unique_min, pop_unique_max))
    sys.stdout.write('wins for R: %.0f\n' %win_r)
    sys.stdout.write('wins for M: %.0f\n' %win_m)
    sys.stdout.write('wins for I: %.0f\n' %win_i)
    sys.stdout.write('wins for A: %.0f\n' %win_a)
    sys.stdout.write('wins for B: %.0f\n' %win_b)
    sys.stdout.write('wins for E: %.0f\n' %win_e)

if __name__ == "__main__":
    main()


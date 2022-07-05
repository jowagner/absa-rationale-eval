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

def get_synthetic_population(size, interval = 100):
    population = []
    for _ in range(size):
        values = []
        for _ in range(8):
            values.append(random.random())
            values.append(random.random()**2)
        population.append(1000+int(interval*sum(values)/16.0))
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
    add_annotation(annotation, m_q1, 'J-Q1')
    add_annotation(annotation, m_q3, 'J-Q3')
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

    sys.stdout.write('\n== Evenly spaces values ==\n\n')
    sample = get_interpolated_range(n)
    annotation = {}
    annotate_values(sample, annotation)
    print_sample_with_annotation(sample, annotation)

    sys.stdout.write('\n== A skewed distribution ==\n\n')
    sample = get_synthetic_population(n)
    annotation = {}
    annotate_values(sample, annotation)
    print_sample_with_annotation(sample, annotation)

    #sys.exit(0)

    sys.stdout.write('\n== Which method works best on average? ==\n\n')
    
    # TODO: implement methods 3 and 4 from
    #       https://en.wikipedia.org/wiki/Quartile

    # let's count which method works best on average
    repetitions = 1200
    win_r = 0
    win_m = 0
    win_i = 0
    pop_unique_min = None
    pop_unique_max = None
    for _ in range(repetitions):
        if verbose:
            print('\n=== Repetition ==\n\n')
        population = get_synthetic_population(4005, int(sys.argv[2]))  # 5 + 4k with k = 1000
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
        if verbose:
            sys.stdout.write('     \t R  \t M  \t I  \t span\n')
            sys.stdout.write('T(Q1)\t%.2f\t%.2f\t%.2f\t%.2f\n' %(t_r_q1, t_m_q1, t_i_q1, max(t_r_q1, t_m_q1, t_i_q1) - min(t_r_q1, t_m_q1, t_a_q1)))
            sys.stdout.write('T(Q3)\t%.2f\t%.2f\t%.2f\t%.2f\n' %(t_r_q3, t_m_q3, t_i_q3, max(t_r_q3, t_m_q3, t_i_q3) - min(t_r_q3, t_m_q3, t_a_q3)))
            sys.stdout.write('\n')
            sys.stdout.write('pop range: %d to %d\n' %(min(population), max(population)))
            sys.stdout.write('\n')

        sum_r_q1 = 0.0
        sum_r_q3 = 0.0
        sum_m_q1 = 0.0
        sum_m_q3 = 0.0
        sum_i_q1 = 0.0
        sum_i_q3 = 0.0
        n_samples = 1024
        for _ in range(n_samples):
            values = random.choices(population, k = n)
            median = get_median(values)
            r_q1, r_q3 = r_q(values, median)
            m_q1, m_q3 = m_q(values, median)
            i_q1, i_q3 = i_q(values, median)
            sum_r_q1 += r_q1
            sum_r_q3 += r_q3
            sum_m_q1 += m_q1
            sum_m_q3 += m_q3
            sum_i_q1 += i_q1
            sum_i_q3 += i_q3
        avg_r_q1 = sum_r_q1 / float(n_samples)
        avg_r_q3 = sum_r_q3 / float(n_samples)
        avg_m_q1 = sum_m_q1 / float(n_samples)
        avg_m_q3 = sum_m_q3 / float(n_samples)
        avg_i_q1 = sum_i_q1 / float(n_samples)
        avg_i_q3 = sum_i_q3 / float(n_samples)
        if verbose:
            sys.stdout.write('     \t R  \t error \t M  \t error \t I  \t error \twinner\n')
        for q, t_r_q, avg_r_q, t_m_q, avg_m_q, t_i_q, avg_i_q in [
            (1, t_r_q1, avg_r_q1, t_m_q1, avg_m_q1, t_i_q1, avg_i_q1),
            (3, t_r_q3, avg_r_q3, t_m_q3, avg_m_q3, t_i_q3, avg_i_q3),
        ]:
            error_and_tag = [
                (abs(t_r_q - avg_r_q), random.random(), 'R'),
                (abs(t_m_q - avg_m_q), random.random(), 'M'),
                (abs(t_i_q - avg_i_q), random.random(), 'I'),
            ]
            winner = sorted(error_and_tag)[0][-1]
            if verbose:
                sys.stdout.write('E(Q%d)\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n' %(
                    q,
                    avg_r_q, abs(t_r_q - avg_r_q),
                    avg_m_q, abs(t_m_q - avg_m_q),
                    avg_i_q, abs(t_i_q - avg_i_q),
                    winner,
                ))
            if winner == 'R':
                win_r += 1
            elif winner == 'M':
                win_m += 1
            elif winner == 'I':
                win_i += 1
            else:
                raise ValueError
    
        if verbose:
            sys.stdout.write('\n')
    
    sys.stdout.write('unique size range %d - %d\n' %(pop_unique_min, pop_unique_max))
    sys.stdout.write('wins for R: %d\n' %win_r)
    sys.stdout.write('wins for M: %d\n' %win_m)
    sys.stdout.write('wins for I: %d\n' %win_i)

if __name__ == "__main__":
    main()


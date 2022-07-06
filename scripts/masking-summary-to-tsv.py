#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

from quartiles import BoxPlot

opt_show_stddev_in_appendix = False
opt_number_of_sets = 4
opt_runs_per_set = 3

include_domain_breakdown = False
if len(sys.argv) > 1:
    if len(sys.argv) == 2 and sys.argv[1] in ('--include-domain-breakdown', '--domains'):
        include_domain_breakdown = True
    else:
        raise ValueError('unsupported option(s)')

expected_total_runs = opt_number_of_sets * opt_runs_per_set

example = """
==> c-f-1-1/stdout-training-with-local-aio.txt <==

Summary:
SeqB                  Q                     Overall               Laptop                Restaurant            Description
Full                  0                     83.834135532          81.930691004          85.686725378          Sun et al. QA-M
SE                    0                     81.730771065          79.084157944          84.297835827          Sun et al. QA-M
Other                 0                     63.822120428          56.064355373          70.640432835          Sun et al. QA-M

==> c-f-1-2/stdout-training-with-local-aio.txt <==

Summary:
SeqB                  Q                     Overall               Laptop                Restaurant            Description
Full     """

trshort2sortkey = {
    'f': 'tr=Full',
    'n': 'tr=None',
    's': 'tr=SE/R',
    'o': 'tr=Y_Other',
    'a': 'tr=Z_Concat',
}

data = {}
run = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    elif line.startswith('==>'):
        fields = line.replace('/', ' ').split()
        #     [1]     [2]
        # ==> c-f-1-2/stdout-training-with-local-aio.txt <==
        assert len(fields) == 4
        folder = fields[1]
        filename = fields[2]
        fields = folder.split('-')
        assert len(fields) == 4
        assert fields[0] == 'c'
        set_index = int(fields[2])-1
        run_in_set = int(fields[3])
        run = opt_runs_per_set * set_index + run_in_set
        tr = trshort2sortkey[fields[1]]
        if filename in ('stdout-training-with-sea-aio.txt', 'stdout.txt'):
            m_type = 'tab2-SE'
        elif 'with-union-aio' in filename:
            m_type = 'tab2-U-SE'
        elif filename.startswith('stdout-training-with-RND'):
            fields = filename.replace('-', ' ').split()
            aio_name = fields[3].replace('.', ' ').split()[0]
            m_type = 'tab4-' + aio_name
        elif filename.startswith('stdout-training-with-'):
            fields = filename.replace('-', ' ').split()
            aio_name = fields[3].replace('.', ' ').split()[0]
            m_type = 'tab3-' + aio_name
        elif filename == 'stdout-training-with-local-aio.txt':
            m_type = 'tab5-old-R'
        else:
            raise ValueError('unsupported path %s/%s' %(folder, filename))
    elif line.startswith('SeqB'):
        fields = line.split('\t')
        # check column header
        assert fields[2] == 'Overall'
        assert fields[3] == 'Laptop'
        assert fields[4] == 'Restaurant'
        assert fields[5].rstrip() == 'Description'
    elif 'Sun et al. QA-M' in line:
        if line.startswith('Full'):
            te = 'Full'
        elif line.startswith('None'):
            te = 'None'
        elif line.startswith('SE'):
            te = 'SE/R'
        elif line.startswith('Other'):
            te = 'Z-CompSE/R'
        else:
            raise ValueError(line)
        for index, domain in enumerate('Overall Laptop Restaurant'.split()):
            score = line.split('\t')[2+index]
            row = []
            row.append(m_type)
            row.append(tr)
            row.append(domain)
            row.append(te)
            row.append('%d' %run)
            row.append(score)
            sys.stdout.write('\t'.join(row))
            sys.stdout.write('\n')
            key = (m_type, tr, domain, te, run)
            data[key] = float(score)

# create latex tables

f = open('results-masking-diagonal.tex', 'wt')
f.write(r"""% Table with masking results, diagonal of results from Appendix

\begin{table}
    \small
    \centering
    \begin{tabular}{l|rr}
    %\hline
    \textbf{Input} & \textbf{Accuracy} & \textbf{Input} & \textbf{Accuracy}  \\
    \hline
""")

cell_to_scores = {}


def get_cell_scores(m_type, tr, domain, te, extra_sets = 0):
    global opt_runs_per_set
    global expected_total_runs
    global data
    scores = []
    total_runs = expected_total_runs + opt_runs_per_set * extra_sets
    for run in range(1, total_runs+1):
        key = (m_type, tr, domain, te, run)
        if key in data:
            scores.append(data[key])
    return scores

def get_cell_content(m_type, tr, domain, te, show_stddev = True, extra_sets = 0):
    global opt_runs_per_set
    global expected_total_runs
    global data
    scores = get_cell_scores(m_type, tr, domain, te, extra_sets)
    total_runs = expected_total_runs + opt_runs_per_set * extra_sets
    if len(scores) < total_runs:
        sys.stderr.write('Warning: Expected %d runs for %r but only found %d\n' %(
            total_runs, (m_type, tr, domain, te), len(scores),
        ))
        #for key in data:
        #    if key[:4] == (m_type, tr, domain, te):
        #        sys.stderr.write('\t[%r]\t%.9f\n' %(key, data[key]))
        return '--.-   -  -.- '
    avg_score = sum(scores)/float(len(scores))
    if not show_stddev:
        return '%.1f' %avg_score
    sq_errors = []
    for score in scores:
        error = avg_score - score
        sq_errors.append(error**2)
    n = len(scores)
    #std_dev = (sum(sq_errors)/float(n))**0.5                   # Population std dev
    #std_dev = (sum(sq_errors)/(n-1.0))**0.5                    # Simple sample std dev
    #std_dev = (sum(sq_errors)/(n-1.5))**0.5                    # Approximate std dev
    std_dev = (sum(sq_errors)/(n-1.5+1.0/(8.0*(n-1.0))))**0.5  # More accurate std dev
    if std_dev < 0.7:
        # usable for box plot test
        cell_to_scores[(m_type, tr, domain, te)] = scores
    return r'%.1f $\pm %.1f$' %(avg_score, std_dev)

is_first = True
for domain, maj_acc, baseline_acc in [
    ('Laptop',     60.0, get_cell_content('tab2-SE', 'tr=Full', 'Laptop',     'Full', extra_sets = 4)),
    ('Restaurant', 71.1, get_cell_content('tab2-SE', 'tr=Full', 'Restaurant', 'Full', extra_sets = 4)),
    ('Overall',    65.8, get_cell_content('tab2-SE', 'tr=Full', 'Overall',    'Full', extra_sets = 4)),
]:
    if not include_domain_breakdown and domain != 'Overall':
        continue
    if not is_first:
        f.write(r"""    \multicolumn{3}{l}{} \\
""")
    is_first = False
    f.write(r'    \textbf{Full}                      & ')
    f.write(get_cell_content('tab2-SE', 'tr=Full', domain, 'Full', extra_sets = 4))
    f.write(r' & \textbf{None}                            & ')
    f.write(get_cell_content('tab2-SE', 'tr=None', domain, 'None', extra_sets = 4))
    f.write(r' \\')
    f.write('\n')
    for m_type, mask_title_left in [
        ('tab2-SE',   'SE'),
        ('tab2-U-SE', 'U-SE'),
        (None, r'\hline'),
        ('tab3-L25',  'R\\textsubscript{IG}@.25'),
        ('tab3-L50',  'R\\textsubscript{IG}@.5'),
        ('tab3-L75',  'R\\textsubscript{IG}@.75'),
        (None, r'\hline'),
        ('tab3-P25',  'R\\textsubscript{APG}@.25'),
        ('tab3-P50',  'R\\textsubscript{APG}@.5'),
        ('tab3-P75',  'R\\textsubscript{APG}@.75'),
        (None, r'\hline'),
        (None, r'\multicolumn{4}{l}{LIME with 2500 samples, abs score of predicted class} \\'),
        ('tab3-M25',  'R\\textsubscript{LIME}@.25'),
        ('tab3-M50',  'R\\textsubscript{LIME}@.5'),
        ('tab3-M75',  'R\\textsubscript{LIME}@.75'),
        (None, r'\hline'),
        (None, r'\multicolumn{4}{l}{LIME with 2500 samples, plain score of predicted class} \\'),
        ('tab3-N25',  'R\\textsubscript{LIME}@.25'),
        ('tab3-N50',  'R\\textsubscript{LIME}@.5'),
        ('tab3-N75',  'R\\textsubscript{LIME}@.75'),
        (None, r'\hline'),
        (None, r'\multicolumn{4}{l}{LIME with 2500 samples, any support for predicted class} \\'),
        ('tab3-S25',  'R\\textsubscript{LIME}@.25'),
        ('tab3-S50',  'R\\textsubscript{LIME}@.5'),
        ('tab3-S75',  'R\\textsubscript{LIME}@.75'),
        (None, r'\hline'),
        (None, r'\multicolumn{4}{l}{LIME with 2500 samples, max(abs(s1),abs(s2),abs(s3))} \\'),
        ('tab3-X25',  'R\\textsubscript{LIME}@.25'),
        ('tab3-X50',  'R\\textsubscript{LIME}@.5'),
        ('tab3-X75',  'R\\textsubscript{LIME}@.75'),
        (None, r'\hline'),
        (None, r'\multicolumn{4}{l}{Saliency map with random values (same map for all 6} \\'),
        (None, r'\multicolumn{4}{l}{settings, \eg @.25 $\subseteq$ @.5; 12 maps, one for each run)} \\'),
        ('tab4-RND25',  'R\\textsubscript{RAND}@.25'),
        ('tab4-RND50',  'R\\textsubscript{RAND}@.5'),
        ('tab4-RND75',  'R\\textsubscript{RAND}@.75'),
        (None, r'\hline'),
    ]:
        f.write('    ')
        if not m_type:
            # print special line
            f.write(mask_title_left)
            f.write('\n')
            continue
        for is_first_column, tr, te, mask_title in [
            (True,  'tr=SE/R',    'SE/R',       mask_title_left),
            (False, 'tr=Y_Other', 'Z-CompSE/R', '$\\neg$' + mask_title_left),
        ]:
            if not is_first_column:
                col_width = 32
                f.write('& ')
            else:
                col_width = 26
            gap = (col_width - len(mask_title)) * ' '
            f.write(r'\textbf{%s}%s' %(mask_title, gap))
            f.write(r'& %s ' %get_cell_content(m_type, tr, domain, te))
        f.write(r'\\')
        f.write('\n')
    f.write(r"""    \multicolumn{3}{l}{Test set: %(domain)s} \\
    \multicolumn{3}{l}{Majority baseline: %(maj_acc).1f} \\
    \hline
""" %locals())
f.write(r"""    \end{tabular}
    \caption{Test set accuracy (x100, average and standard deviation over twelve runs)
             and effect of restricting input to SEs,
             the union of all SEs where a sentence has multiple opinions (U-SE),
             rationales based on integrated gradients (R\textsubscript{IG}),
             rationales based on absolute point gradients (R\textsubscript{APG}),
             rationales based on LIME scores (R\textsubscript{LIME}),
             random tokens (R\textsubscript{RAND}) and
             masking all other tokens ($\neg$)
             for 25\%, 50\% and 75\% lengths.
             ``None'' masks the review sentence completely.
             The review domain, aspect entity type, aspect attribute
             and sentence length are available to all classifiers.}
    \label{tab:masking:rationales-diagonal}
\end{table}
% eof
""")
f.close()
#            The majority baselines ``all positive''
#            have 60.0\% and 71.1\% accuracy respectively.}


# Appendix tables

# SE and U-SE

for m_type, mask_filename, mask_title in [
    ('tab2-SE',    '0SE',   'SE'),
    ('tab2-U-SE',  '0U-SE', 'U-SE'),
    ('tab3-L25',   'L25',   'R\\textsubscript{IG}@.25'),
    ('tab3-L50',   'L50',   'R\\textsubscript{IG}@.5'),
    ('tab3-L75',   'L75',   'R\\textsubscript{IG}@.75'),
    ('tab3-P25',   'P25',   'R\\textsubscript{APG}@.25'),
    ('tab3-P50',   'P50',   'R\\textsubscript{APG}@.5'),
    ('tab3-P75',   'P75',   'R\\textsubscript{APG}@.75'),
    ('tab3-N25',   'N25',   'R\\textsubscript{LIME}@.25'),
    ('tab3-N50',   'N50',   'R\\textsubscript{LIME}@.5'),
    ('tab3-N75',   'N75',   'R\\textsubscript{LIME}@.75'),
    ('tab4-RND25', 'RND25', 'R\\textsubscript{RAND}@.25'),
    ('tab4-RND50', 'RND50', 'R\\textsubscript{RAND}@.5'),
    ('tab4-RND75', 'RND75', 'R\\textsubscript{RAND}@.75'),
]:
    f = open('results-masking-%s.tex' %mask_filename, 'wt')
    f.write(r"""%% Table with %(mask_title)s masking results

\begin{table}
    \centering
    \begin{tabular}{l|rrrr}
    \textbf{Model} & \multicolumn{3}{c}{\textbf{Test Accuracy}} \\
                   & \textbf{Full}
                   & \textbf{None}
                   & \textbf{%(mask_title)s}
                   & \textbf{$\neg$%(mask_title)s} \\
    \hline
""" %locals())

    is_first = True
    for domain, maj_acc, in [
        ('Laptop',     60.0),
        ('Restaurant', 71.1),
        ('Overall',    65.8),
    ]:
        if not include_domain_breakdown and domain != 'Overall':
            continue
        if not is_first:
            f.write(r"""    \multicolumn{3}{l}{} \\
""")
        is_first = False
        f.write(r"""    \multicolumn{3}{l}{Test set: %(domain)s} \\
    %%\multicolumn{3}{l}{Majority baseline: %(maj_acc).1f} \\
    \hline
""" %locals())

        for tr, tr_title in [
            ('tr=Full',     'Full'),
            ('tr=None',     'None'),
            ('tr=SE/R',     mask_title),
            ('tr=Y_Other',  '$\\neg$' + mask_title),
            ('tr=Z_Concat', 'Concat'),
        ]:
            gap = (25 - len(tr_title)) * ' '
            f.write(r'    \textbf{%s}%s' %(tr_title, gap))
            for te in [
                'Full',
                'None',
                'SE/R',
                'Z-CompSE/R',
            ]:
                if m_type == 'tab2-SE' and tr in ('tr=Full', 'tr=None'):
                    extra_sets = 4
                else:
                    extra_sets = 0
                try:
                    f.write('& %s ' %get_cell_content(
                        m_type, tr, domain, te,
                        show_stddev = opt_show_stddev_in_appendix,
                        extra_sets = extra_sets,
                    ))
                except ValueError:
                    f.write('& -- ')
            f.write(r'\\')
            f.write('\n')
    f.write(r"""    \end{tabular}
    \caption{Out of distrubtion settings for %(mask_title)s.}
    \label{tab:masking:rationales-%(mask_title)s}
\end{table}
%% eof
""" %locals())
    f.close()
    #            The majority baselines ``all positive''
    #            have 60.0\% and 71.1\% accuracy respectively.}

# fake accuracy scores around 80% accuracy based on observed
# variation

n = 1664
n80 = int(0.5+0.8*n)

f = open('centred-accuracies.txt', 'wt')
for key in cell_to_scores:
    correct = []
    total = 0
    for score in cell_to_scores[key]:
        count = int(0.5+n*score/100.0)  # score is accuracy * 100
        correct.append(count)
        total += count
    avg_count = total // len(correct)
    for count in correct:
        count = n80 + count - avg_count
        accuracy = 100.0*count/float(n)
        if 78.0 <= accuracy <= 82.0:
            f.write('%.9f\n' %accuracy)
f.close()

# compile data for box plots

for domain in 'Laptop Restaurant Overall'.split():
    if not include_domain_breakdown and domain != 'Overall':
        continue
    f = open('box-plot-full-rnd-none.tsv', 'wt')
    
    boxplots = []
    boxplots.append(('None', BoxPlot(
        get_cell_scores('tab2-SE', 'tr=None', domain, 'None', extra_sets = 4),
    )))

    boxplots.append(('RND25', BoxPlot(
        get_cell_scores('tab4-RND25', 'tr=SE/R',       domain, 'SE/R' ) + \
        get_cell_scores('tab4-RND75', 'tr=Z-CompSE/R', domain, 'Z-CompSE/R' ),
    )))
    boxplots.append(('RND50', BoxPlot(
        get_cell_scores('tab4-RND50', 'tr=SE/R',       domain, 'SE/R' ) + \
        get_cell_scores('tab4-RND50', 'tr=Z-CompSE/R', domain, 'Z-CompSE/R' ),
    )))
    boxplots.append(('RND75', BoxPlot(
        get_cell_scores('tab4-RND75', 'tr=SE/R',       domain, 'SE/R' ) + \
        get_cell_scores('tab4-RND25', 'tr=Z-CompSE/R', domain, 'Z-CompSE/R' ),
    )))
    boxplots.append(('Full', BoxPlot(
        get_cell_scores('tab2-SE', 'tr=Full', domain, 'Full', extra_sets = 4),
    )))
    header = []
    header.append('Attribute')
    for bp_name, _ in boxplots:
        header.append(bp_name)
    f.write('\t'.join(header))
    f.write('\n')
    for attr_name in 'B Q1 M Q3 T'.split():
        row = []
        row.append(attr_name)
        for bp_name, boxplot in boxplots:
            row.append('%.9f' %(boxplot[attr_name]))
        f.write('\t'.join(row))
        f.write('\n')
    outlier_index = 0
    while True:
        found_outlier = False
        row = []
        row.append('O_%d' %(outlier_index + 1))
        for bp_name, boxplot in boxplots:
            try:
                outlier = '%.9f' %(boxplot[('O', outlier_index)])
                found_outlier = True
            except IndexError:
                outlier = ''
            row.append(outlier)
        if not found_outlier:
            break
        f.write('\t'.join(row))
        f.write('\n')
        outlier_index += 1
    f.close()

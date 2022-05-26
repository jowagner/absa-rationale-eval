#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

opt_show_stddev_in_appendix = False

include_domain_breakdown = False
if len(sys.argv) > 1:
    if len(sys.argv) == 2 and sys.argv[1] in ('--include-domain-breakdown', '--domains'):
        include_domain_breakdown = True
    else:
        raise ValueError('unsupported option(s)')

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
        run = 3*(int(fields[2])-1) + int(fields[3])
        tr = trshort2sortkey[fields[1]]
        if filename in ('stdout-training-with-sea-aio.txt', 'stdout.txt'):
            m_type = 'tab2-SE'
        elif 'with-union-aio' in filename:
            m_type = 'tab2-U-SE'
        elif filename.startswith('stdout-training-with-L'):
            fields = filename.replace('-', ' ').split()
            aio_name = fields[3].replace('.', ' ').split()[0]
            m_type = 'tab3-' + aio_name
        elif filename.startswith('stdout-training-with-RND'):
            fields = filename.replace('-', ' ').split()
            aio_name = fields[3].replace('.', ' ').split()[0]
            m_type = 'tab4-' + aio_name
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
    %\small
    \centering
    \begin{tabular}{l|rr}
    %\hline
    \textbf{Input} & \textbf{Accuracy} & \textbf{Input} & \textbf{Accuracy}  \\
    \hline
""")

def get_cell_content(m_type, tr, domain, te, show_stddev = True):
    scores = []
    for run in range(1,10):
        key = (m_type, tr, domain, te, run)
        if key in data:
            scores.append(data[key])
    if len(scores) != 9:
        #raise ValueError('Expected 9 scores, got %r for m_type %r, tr %r, domain %r, te %r and run %r' %(scores, m_type, tr, domain, te, run))
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
    return r'%.1f $\pm %.1f$' %(avg_score, std_dev)

is_first = True
for domain, maj_acc, baseline_acc in [
    ('Laptop',     60.0, get_cell_content('tab2-SE', 'tr=Full', 'Laptop', 'Full')),
    ('Restaurant', 71.1, get_cell_content('tab2-SE', 'tr=Full', 'Restaurant', 'Full')),
    ('Overall',    65.8, get_cell_content('tab2-SE', 'tr=Full', 'Overall', 'Full')),
]:
    if not include_domain_breakdown and domain != 'Overall':
        continue
    if not is_first:
        f.write(r"""    \multicolumn{3}{l}{} \\
""")
    is_first = False
    f.write(r"""    \multicolumn{3}{l}{Test set: %(domain)s} \\
    \multicolumn{3}{l}{Majority baseline: %(maj_acc).1f} \\
    \hline
""" %locals())
    f.write(r'    \textbf{Full}        & %s & \textbf{None}        & %s \\' %(
        get_cell_content('tab2-SE', 'tr=Full', domain, 'Full'),
        get_cell_content('tab2-SE', 'tr=None', domain, 'None'),
    ))
    f.write('\n')
    for m_type, mask_title_left in [
        ('tab2-SE',   'SE'),
        ('tab2-U-SE', 'U-SE'),
        (None, None),            # = hline separator
        ('tab3-L25',  'R@.25'),
        ('tab3-L50',  'R@.5'),
        ('tab3-L75',  'R@.75'),
        (None, None),            # = hline separator
        ('tab4-RND25',  'A@.25'),
        ('tab4-RND50',  'A@.5'),
        ('tab4-RND75',  'A@.75'),
        (None, None),            # = hline separator
    ]:
        f.write('    ')
        if not m_type:
            f.write(r'\hline')
            f.write('\n')
            continue
        for is_first_column, tr, te, mask_title in [
            (True,  'tr=SE/R',    'SE/R',       mask_title_left),
            (False, 'tr=Y_Other', 'Z-CompSE/R', '$\\neg$' + mask_title_left),
        ]:
            if not is_first_column:
                f.write('& ')
            gap = (12 - len(mask_title)) * ' '
            f.write(r'\textbf{%s}%s' %(mask_title, gap))
            f.write(r'& %s ' %get_cell_content(m_type, tr, domain, te))
        f.write(r'\\')
        f.write('\n')
f.write(r"""    \end{tabular}
    \caption{Test set accuracy (x100, average and standard deviation over nine runs)
             and effect of restricting input to sentiment expressions (SE),
             the union of all SEs where a sentence has multiple opinions (U-SE),
             rationales (R), random tokens (A) and
             masking all other tokens ($\neg$SE, $\neg$R and $\neg$A)
             for 25\%, 50\% and 75\% lengths.
             ``None'' masks the review sentence completely. Only the
             review domain, aspect entity type, aspect attribute and sentence
             length (via the number of ``[MASK]'' tokens) are available to
             the classifier in this setting.}
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
    ('tab3-L25',   'L25',   'R@.25'),
    ('tab3-L50',   'L50',   'R@.5'),
    ('tab3-L75',   'L75',   'R@.75'),
    ('tab4-RND25', 'RND25', 'A@.25'),
    ('tab4-RND50', 'RND50', 'A@.5'),
    ('tab4-RND75', 'RND75', 'A@.75'),
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
    \multicolumn{3}{l}{Majority baseline: %(maj_acc).1f} \\
    \hline
""" %locals())

        for tr, tr_title in [
            ('tr=Full',    'Full'),
            ('tr=None',    'None'),
            ('tr=SE/R',    mask_title),
            ('tr=Y_Other', '$\\neg$' + mask_title),
            ('tr=Z_Concat', 'Concat'),
        ]:
            gap = (11 - len(tr_title)) * ' '
            f.write(r'    \textbf{%s}%s' %(tr_title, gap))
            for te in [
                'Full',
                'None',
                'SE/R',
                'Z-CompSE/R',
            ]:
                try:
                    f.write('& %s ' %get_cell_content(
                        m_type, tr, domain, te,
                        show_stddev = opt_show_stddev_in_appendix
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

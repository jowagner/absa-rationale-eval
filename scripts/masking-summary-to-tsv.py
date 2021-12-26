#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

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
        if filename == 'stdout.txt':
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
    \textbf{Mask} & \multicolumn{2}{c}{\textbf{Training and Test Setting}} \\
                   & \textbf{SE/R/A} & \textbf{$\neg$SE/$\neg$R/$\neg$A} \\
    \hline
""")

def get_cell_content(m_type, tr, domain, te):
    scores = []
    for run in range(1,10):
        key = (m_type, tr, domain, te, run)
        if key in data:
            scores.append(data[key])
    assert len(scores) == 9
    avg_score = sum(scores)/float(len(scores))
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

for domain, maj_acc, baseline_acc in [
    ('Laptop',     60.0, get_cell_content('tab2-SE', 'tr=Full', 'Laptop', 'Full')),
    ('Restaurant', 71.1, get_cell_content('tab2-SE', 'tr=Full', 'Restaurant', 'Full')),
    ('Overall',    65.8, get_cell_content('tab2-SE', 'tr=Full', 'Overall', 'Full')),
]:
    if domain != 'Laptop':
        f.write(r"""    \multicolumn{3}{l}{} \\
""")
    f.write(r"""    \multicolumn{3}{l}{Test set: %(domain)s} \\
    \multicolumn{3}{l}{Majority baseline: %(maj_acc).1f} \\
    \multicolumn{3}{l}{Unmasked (Full): %(baseline_acc)s} \\
    \hline
""" %locals())
    for m_type, mask_title in [
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
        if not m_type:
            f.write(r'    \hline')
            f.write('\n')
            continue
        gap = (5 - len(mask_title)) * ' '
        f.write(r'    \textbf{%s}%s' %(mask_title, gap))
        for tr, te in [
            #('tr=Full',    'Full'),
            ('tr=SE/R',    'SE/R'),
            ('tr=Y_Other', 'Z-CompSE/R'),
        ]:
            f.write(r'& %s ' %get_cell_content(m_type, tr, domain, te))
        f.write(r'\\')
        f.write('\n')
f.write(r"""    \end{tabular}
    \caption{Test set accuracy (x100, average and standard deviation over nine runs)
             and effect of restricting input to sentiment expressions (SE),
             the union of all SEs where a sentence has multiple opinions (U-SE),
             rationales (R), random tokens (A) and
             masking all other tokens ($\neg$SE, $\neg$R, and $\neg$A)
             for 25\%, 50\% and 75\% lengths.}
    \label{tab:masking:rationales-diagonal}
\end{table}
% eof
""")
#            The majority baselines ``all positive''
#            have 60.0\% and 71.1\% accuracy respectively.}

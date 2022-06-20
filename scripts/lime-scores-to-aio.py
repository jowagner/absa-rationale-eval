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
opt_sets    = [('tr', 'train'), ('te', 'test')]
opt_domains = 'laptop restaurant'.split()
opt_thresholds = [25.0, 50.0, 75.0]
opt_wordcloud  = True

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
    elif option == '--thresholds':
        opt_thresholds = []
        for item in sys.argv[1].split(':'):
            opt_thresholds.append(float(item))
        del sys.argv[1]
    elif option == '--no-wordcloud':
        opt_wordcloud = False
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

#opt_verbose = False
#opt_workdir = './'
#opt_prefix  = 'lime-scores'
#opt_sets    = [('tr', 'train'), ('te', 'test')]
#opt_domains = 'laptop restaurant'.split()
#opt_thresholds = [25.0, 50.0, 75.0]
#opt_wordcloud  = True

def abs_score_of_predicted_class(scores):


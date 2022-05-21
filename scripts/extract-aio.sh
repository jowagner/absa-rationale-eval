#!/bin/bash

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# this script assumes that the training script has last been
# run with get_training_saliencies = True

# to be run from the top-level project folder containing the c-?-?-?
# folders and the scripts folder

LOGNAME=saliency-morewio-xfw-stdout.txt

for L in L25 L50 L75 ; do
    for D in laptop restaurant ; do
        for I in c-f-?-? ; do
            echo $L $D $I
            grep -E "^"${L}"\s"${D}"\s(training|dev)\s" -A 1 $I/$LOGNAME \
                | grep -v -E "^--$" \
                | tee $I/train-${D}-${L}-wcloud.tsv \
                | cut -f4,5,6 \
                | scripts/sort-aio-by-id.py \
                | cut -f2,3 > $I/train-${D}-${L}.aio
            grep -E "^"${L}"\s"${D}"\stest\s" -A 1 $I/$LOGNAME \
                | grep -v -E "^--$" \
                | tee $I/test-${D}-${L}-wcloud.tsv \
                | cut -f4,5,6 \
                | scripts/sort-aio-by-id.py \
                | cut -f2,3 > $I/test-${D}-${L}.aio
            # only keep wcloud.tsv for L50
            rm -f $I/train-${D}-L?5-wcloud.tsv
done ; done ; done


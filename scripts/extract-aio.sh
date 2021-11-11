#!/bin/bash

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# this script assumes that the training script has last been
# run with get_training_saliencies = True

for L in L20 L40 L50 ; do
    for D in laptop restaurant ; do
        for I in c-f-?-? ; do
            echo $L $D $I
            grep -E "^"${L}"\s"${D}"\s(training|dev)\s" -A 1 $I/saliency-allwio-stdout.txt \
                | grep -v -E "^--$" \
                | cut -f4,5,6 \
                | scripts/sort-aio-by-id.py \
                | cut -f2,3 > $I/train-${D}-${L}.aio
            grep -E "^"${L}"\s"${D}"\stest\s" -A 1 $I/saliency-allwio-stdout.txt \
                | grep -v -E "^--$" \
                | cut -f4,5,6 \
                | scripts/sort-aio-by-id.py \
                | cut -f2,3 > $I/test-${D}-${L}.aio
done ; done ; done

#               | tee $I/train-${D}-${L}-step1.tmp \
#               | tee $I/train-${D}-${L}-step2.tmp \
#               | tee $I/test-${D}-${L}-step1.tmp \
#               | tee $I/test-${D}-${L}-step2.tmp \

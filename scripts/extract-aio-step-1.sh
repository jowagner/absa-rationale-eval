#!/bin/bash

# this script assumes that the training script has last been
# run with get_training_saliencies = True and that the output
# for training data does not include the items selected for
# the dev set, hence "step 1"

for L in L20 L40 L50 ; do
    for D in laptop restaurant ; do
        for I in c-f-?-? ; do
            echo $L $D $I
            grep -E "^"${L}"\s"${D}"\s" -A 1 $I/saliency-stdout.txt \
                | grep -v -E "^--$" \
                | cut -f3,4 > $I/train-${D}-${L}.aio
done ; done ; done

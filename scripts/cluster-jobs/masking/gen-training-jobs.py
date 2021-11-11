#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys


local_aio = True

for tr_task_short, tr_task_long in [
    ('f', 'Full'),
    ('s', 'SE'),
    ('o', 'Other'),
    ('a', 'All'),
]:
    for set_rank in (1,2,3):
        with open('run-train-c-%s-%d1-to-%d3.job' %(tr_task_short, set_rank, set_rank), 'w') as f:
            f.write("""#!/bin/bash

#SBATCH -p compute       # which partition to run on
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J ab-%(tr_task_short)s%(set_rank)d-2h    # name for the job
#SBATCH --mem=47000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH -N 1-1

TR_TASK=%(tr_task_long)s
TR_TASK_SHORT=%(tr_task_short)s
SET=%(set_rank)d
L=L50
""" %locals())
            if local_aio:
                f.write("""
DESC=training-with-local-aio
""")
            else:
                f.write("""
DESC=training
""")

            f.write("""
PRJ_DIR=${HOME}/sentiment/absa-rationale-eval
cd $PRJ_DIR
source venv-pytorch/bin/activate

for RUN in 1 2 3 ; do
    echo "== Run $RUN =="
    date
    cd $PRJ_DIR
    MODEL_DIR=c-${TR_TASK_SHORT}-${SET}-${RUN}
    mkdir $MODEL_DIR
    cd $MODEL_DIR
    mv best-model-weights-only.ckpt $(mktemp -u best-model-weights-only-XXXXXXXXXXXX.ckpt)
    ln -s ../data
""")
            if local_aio:
                f.write("""
    mkdir local-aio
    hostname >> local-aio/prep.start
    date     >> local-aio/prep.start
    touch       local-aio/prep.start
    for D in laptop restaurant ; do
        for T in train test ; do
            cp ../c-f-${SET}-${RUN}/${T}-${D}-${L}.aio local-aio/${T}.${D}.aio
        done
    done
""")

            f.write("""
    hostname        >> ${DESC}.start
    date            >> ${DESC}.start
    echo $MODEL_DIR >> ${DESC}.start
    echo $TR_TASK   >> ${DESC}.start
    touch              ${DESC}.start
""")

            if local_aio:
                f.write("""
    ../scripts/train-classifier.py --local-aio 2${SET}${RUN} train $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""")
            else:
                f.write("""
    ../scripts/train-classifier.py ${SET}${RUN} train $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""")

            f.write("""
    touch ${DESC}.end
    date
    echo "done"
done
""")

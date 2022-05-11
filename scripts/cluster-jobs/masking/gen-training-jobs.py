#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

local_aio = True

#fhp = open('../../hparams.txt', 'rt')
#while True:
# line = fhp.readline()
# if not line:
#     break
# # print(seed, lr1, lr2, fre, vbs, neps[0])
# fields = line.split()
hparam = 101  # int(fields[0])
lr1 =     10  # int(fields[1])
lr2 =     30  # int(fields[2])
fre =      0  # int(fields[3])
vbs =     64  # int(fields[4])
nep =     10  # int(fields[5])

for tr_task_short, tr_task_long, hours in [
     ('f', 'Full',  int(1+3*0.05*nep)),
     ('s', 'SE',    int(1+3*0.05*nep)),
     ('o', 'Other', int(1+3*0.05*nep)),
     ('a', 'All',   int(1+3*0.15*nep)),
]:
  for aio_name, local_aio, save_as in [
     ('sea', False, 'best-sea.ckpt'),
     ('L25', True,  'best-L25.ckpt'),
     ('L50', True,  'best-L50.ckpt'),
     ('L75', True,  'best-L75.ckpt'),
     ('union', True, 'best-union.ckpt'),
     ('RND25', True, 'best-RND25.ckpt'),
     ('RND50', True, 'best-RND50.ckpt'),
     ('RND75', True, 'best-RND75.ckpt'),
  ]:
    for set_rank in (1,2,3):
        with open('run-train-c-%s-%s-%d1-to-%d3.job' %(   #-hp-%d.job' %(
            aio_name, tr_task_short, set_rank, set_rank #, hparam
        ), 'w') as f:
            f.write("""#!/bin/bash

#SBATCH -p compute       # which partition to run on
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J ab-%(tr_task_short)s%(set_rank)d-%(hours)dh    # name for the job
#SBATCH --mem=47000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH -N 1-1
#SBATCH --exclude=g103

TR_TASK=%(tr_task_long)s
TR_TASK_SHORT=%(tr_task_short)s
SET=%(set_rank)d
HPARAM=''
L=%(aio_name)s
""" %locals())
            f.write("""
DESC=training-with-%(aio_name)s-aio
""" %locals())

            f.write("""
PRJ_DIR=${HOME}/sentiment/absa-rationale-eval
cd $PRJ_DIR
source venv-pytorch/bin/activate

for RUN in 1 2 3 ; do
    echo "== Run $RUN =="
    date
    cd $PRJ_DIR
    MODEL_DIR_PREFIX=c-${TR_TASK_SHORT}-${SET}-${RUN}
    #MODEL_DIR=${MODEL_DIR_PREFIX}-${HPARAM}
    MODEL_DIR=${MODEL_DIR_PREFIX}
    mkdir $MODEL_DIR
    cd $MODEL_DIR
    mv best-model-weights-only.ckpt $(mktemp -u best-model-weights-only-XXXXXXXXXXXX.ckpt)
    ln -s ../data
""")
            if local_aio:
                f.write("""
    LAIODIR=local-aio-%(aio_name)s
    mkdir $LAIODIR
    hostname >> $LAIODIR/prep.start
    date     >> $LAIODIR/prep.start
    touch       $LAIODIR/prep.start
    for D in laptop restaurant ; do
        for T in train test ; do
""" %locals())
                if aio_name.startswith('L'):
                    f.write("""
            cp ../c-f-${SET}-${RUN}/${T}-${D}-${L}.aio $LAIODIR/${T}.${D}.aio
""" %locals())
                elif aio_name == 'union':
                    f.write("""
            ../scripts/union-aio.py < data/${T}.${D}.aio > $LAIODIR/${T}.${D}.aio
""" %locals())
                elif aio_name.startswith('RND'):
                    rnd_p = float(aio_name[3:]) / 100.0
                    f.write("""
            ../scripts/random-aio.py --seed ${SET}${RUN} %(rnd_p).9f < data/${T}.${D}.aio > $LAIODIR/${T}.${D}.aio
""" %locals())
                else:
                    raise ValueError('unknown aio_name %s' %aio_name)
                f.write("""
        done
    done
""" %locals())

            f.write("""
    hostname        >> ${DESC}.start
    date            >> ${DESC}.start
    echo $MODEL_DIR >> ${DESC}.start
    echo $TR_TASK   >> ${DESC}.start
    touch              ${DESC}.start
""")

            if local_aio:
                f.write("""
    ../scripts/train-classifier.py --aio-prefix $LAIODIR/ --save-model-as %(save_as)s --lr1 %(lr1)d --lr2 %(lr2)d --fre %(fre)d --vbs %(vbs)d --epochs %(nep)d --trdev-seed ${SET}${RUN} ${HPARAM}${SET}${RUN} train $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""" %locals())
            else:
                f.write("""
    ../scripts/train-classifier.py --save-model-as %(save_as)s --lr1 %(lr1)d --lr2 %(lr2)d --fre %(fre)d --vbs %(vbs)d --epochs %(nep)d --trdev-seed ${SET}${RUN} ${HPARAM}${SET}${RUN} train $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""" %locals())

            f.write("""
    rm -rf lightning_logs
    date >> ${DESC}.end
    touch ${DESC}.end
    cd $PRJ_DIR
    #./pick-model.py ${MODEL_DIR_PREFIX} ${DESC}
    date
    echo "done"
done
""")

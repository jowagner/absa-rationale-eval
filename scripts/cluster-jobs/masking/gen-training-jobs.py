#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys

local_aio = True

if len(sys.argv) > 1 and sys.argv[1] == '--eval':
    mode_code  = 'e'
    mode_short = 'eval'
    mode_long  = 'evaluation'
    model_action = 'load-model-from'
else:
    mode_code  = 't'
    mode_short = 'train'
    mode_long  = 'training'
    model_action = 'save-model-as'

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

gpus = [
    ('rtx6000',  16, ''),
    #('rtxa6000',  32, ''),  # cuda error with current cuda environment
    ('rtx3090',  16, ''),
    #('titanv',   6, '#SBATCH --reservation=themea'),  # issue with cuda env
    ('rtx2080ti', 8, ''),
]

for tr_task_short, tr_task_long, runs_per_set, hours_per_epoch in [
     ('f', 'Full',   3, 0.149),
     ('s', 'SE',     3, 0.149),
     ('o', 'Other',  3, 0.149),
     ('n', 'None',   3, 0.149),
     ('a', 'All',   12, 0.149),
]:
 if mode_short == 'eval':
  hours_per_epoch = 0.0
 hours = int(1+runs_per_set*hours_per_epoch*nep)
 max_minutes = int(45+1.2*runs_per_set*hours_per_epoch*60*nep)
 for gpu, batchsize, more_sbatch in gpus:
  if not os.path.exists(gpu):
      os.makedirs(gpu)
  for aio_name, local_aio, save_as in [
     ('sea', False, 'best-sea.ckpt'),
     ('L25', True,  'best-L25.ckpt'),
     ('L50', True,  'best-L50.ckpt'),
     ('L75', True,  'best-L75.ckpt'),
     ('union', True, 'best-union.ckpt'),
     ('RND25', True, 'best-RND25.ckpt'),
     ('RND50', True, 'best-RND50.ckpt'),
     ('RND75', True, 'best-RND75.ckpt'),
     ('P25', True,  'best-P25.ckpt'),
     ('P50', True,  'best-P50.ckpt'),
     ('P75', True,  'best-P75.ckpt'),
     ('M25', True,  'best-M25.ckpt'),
     ('M50', True,  'best-M50.ckpt'),
     ('M75', True,  'best-M75.ckpt'),
     ('N25', True,  'best-N25.ckpt'),
     ('N50', True,  'best-N50.ckpt'),
     ('N75', True,  'best-N75.ckpt'),
     ('S25', True,  'best-S25.ckpt'),
     ('S50', True,  'best-S50.ckpt'),
     ('S75', True,  'best-S75.ckpt'),
     ('X25', True,  'best-X25.ckpt'),
     ('X50', True,  'best-X50.ckpt'),
     ('X75', True,  'best-X75.ckpt'),
     ('TG1', True,  'best-TG1.ckpt'),
     ('TG2', True,  'best-TG2.ckpt'),
     ('TG3', True,  'best-TG3.ckpt'),
     ('TG4', True,  'best-TG4.ckpt'),
  ]:
    sys.stdout.write('%s %s %s\n' %(tr_task_short, gpu, aio_name))
    if aio_name == 'sea' and tr_task_short in 'fn':
        n_sets = 8
    elif aio_name.startswith('RND') and tr_task_short in 'so':
        n_sets = 8
    else:
        n_sets = 4
    for set_index in range(n_sets):
        set_rank = 1 + set_index
        with open('%s/run-%s-c-%s-%s-%d1-to-%d3.job' %(   #-hp-%d.job' %(
            gpu, mode_short, aio_name, tr_task_short, set_rank, set_rank #, hparam
        ), 'w') as f:
            f.write("""#!/bin/bash

#SBATCH -p compute       # which partition to run on
#SBATCH --gres=gpu:%(gpu)s:1
#SBATCH -J a%(mode_code)s-%(tr_task_short)s%(set_rank)d-%(hours)dh    # name for the job
#SBATCH --mem=47000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH -N 1-1
#SBATCH --exclude=g103
#SBATCH -t %(max_minutes)d
%(more_sbatch)s

TR_TASK=%(tr_task_long)s
TR_TASK_SHORT=%(tr_task_short)s
SET=%(set_rank)d
HPARAM=''
L=%(aio_name)s
""" %locals())
            f.write("""
DESC=%(mode_long)s-with-%(aio_name)s-aio
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
""" %locals())
            if mode_short == 'train':
                f.write("""
    if [ -e %(save_as)s ] ; then
        echo "Found existing model -- skipping"
        continue
    fi
    mv best-model-weights-only.ckpt $(mktemp -u best-model-weights-only-XXXXXXXXXXXX.ckpt)
    ln -s ../data
""" %locals())
            if local_aio:
                f.write("""
    LAIODIR=local-aio-%(aio_name)s
    if [ -e $LAIODIR ] ; then
        echo "Re-using $LAIODIR"
    else
        mkdir $LAIODIR
        hostname >> $LAIODIR/prep.start
        date     >> $LAIODIR/prep.start
        touch       $LAIODIR/prep.start
        for D in laptop restaurant ; do
            for T in train test ; do
""" %locals())
                if aio_name[0] in 'LMNPSTX':
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
    fi
""" %locals())

            f.write("""
    hostname        >> ${DESC}.start
    date            >> ${DESC}.start
    echo $MODEL_DIR >> ${DESC}.start
    echo $TR_TASK   >> ${DESC}.start
    nvidia-smi      >> ${DESC}.start
    echo $SLURM_JOB_ID >> ${DESC}.start
    touch              ${DESC}.start
""")

            if local_aio:
                f.write("""
    ../scripts/train-classifier.py --batch-size %(batchsize)d --aio-prefix $LAIODIR/ --%(model_action)s %(save_as)s --lr1 %(lr1)d --lr2 %(lr2)d --fre %(fre)d --vbs %(vbs)d --epochs %(nep)d --trdev-seed ${SET}${RUN} ${HPARAM}${SET}${RUN} %(mode_short)s $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""" %locals())
            else:
                f.write("""
    ../scripts/train-classifier.py --batch-size %(batchsize)d --%(model_action) %(save_as)s --lr1 %(lr1)d --lr2 %(lr2)d --fre %(fre)d --vbs %(vbs)d --epochs %(nep)d --trdev-seed ${SET}${RUN} ${HPARAM}${SET}${RUN} %(mode_short)s $TR_TASK 2> stderr-${DESC}.txt > stdout-${DESC}.txt
""" %locals())

            f.write("""
    date >> ${DESC}.end
    touch ${DESC}.end
    cd $PRJ_DIR
    #./pick-model.py ${MODEL_DIR_PREFIX} ${DESC}
    date
    echo "done"
done
""")

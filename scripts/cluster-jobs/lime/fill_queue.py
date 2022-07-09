#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020, 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import getpass
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

def get_resource(jobs, job_name_prefix):
    for job_name, resource, _, _, _ in jobs:
        if job_name.startswith(job_name_prefix):
            return resource
    # master and cache jobs
    return None

def main():
    opt_jobs = [
        # job_name, resource, script_name, max_waiting, max_running
        ('lpdr-1-1', 'rtx208', 'run-lime-predict-p6-c-f-11-rtx2080ti.job',   0, 4),
        ('lpdr-1-2', 'rtx208', 'run-lime-predict-p6-c-f-12-rtx2080ti.job',   0, 4),
        ('lpdr-1-3', 'rtx208', 'run-lime-predict-p6-c-f-13-rtx2080ti.job',   0, 4),
        ('lpdr-2-1', 'rtx208', 'run-lime-predict-p6-c-f-21-rtx2080ti.job',   0, 4),
        ('lpdr-2-2', 'rtx208', 'run-lime-predict-p6-c-f-22-rtx2080ti.job',   0, 4),
        ('lpdr-2-3', 'rtx208', 'run-lime-predict-p6-c-f-23-rtx2080ti.job',   0, 4),
        ('lpdr-3-1', 'rtx208', 'run-lime-predict-p6-c-f-31-rtx2080ti.job',   0, 4),
        ('lpdr-3-2', 'rtx208', 'run-lime-predict-p6-c-f-32-rtx2080ti.job',   0, 4),
        ('lpdr-3-3', 'rtx208', 'run-lime-predict-p6-c-f-33-rtx2080ti.job',   0, 4),
        ('lpdr-4-1', 'rtx208', 'run-lime-predict-p6-c-f-41-rtx2080ti.job',   0, 4),
        ('lpdr-4-2', 'rtx208', 'run-lime-predict-p6-c-f-42-rtx2080ti.job',   0, 4),
        ('lpdr-4-3', 'rtx208', 'run-lime-predict-p6-c-f-43-rtx2080ti.job',   0, 4),
        ('lpdq-1-1', 'quadro', 'run-lime-predict-p6-c-f-11-rtx6000.job',     0, 2),
        ('lpdq-1-2', 'quadro', 'run-lime-predict-p6-c-f-12-rtx6000.job',     0, 2),
        ('lpdq-1-3', 'quadro', 'run-lime-predict-p6-c-f-13-rtx6000.job',     0, 2),
        ('lpdq-2-1', 'quadro', 'run-lime-predict-p6-c-f-21-rtx6000.job',     0, 2),
        ('lpdq-2-2', 'quadro', 'run-lime-predict-p6-c-f-22-rtx6000.job',     0, 2),
        ('lpdq-2-3', 'quadro', 'run-lime-predict-p6-c-f-23-rtx6000.job',     0, 2),
        ('lpdq-3-1', 'quadro', 'run-lime-predict-p6-c-f-31-rtx6000.job',     0, 2),
        ('lpdq-3-2', 'quadro', 'run-lime-predict-p6-c-f-32-rtx6000.job',     0, 2),
        ('lpdq-3-3', 'quadro', 'run-lime-predict-p6-c-f-33-rtx6000.job',     0, 2),
        ('lpdq-4-1', 'quadro', 'run-lime-predict-p6-c-f-41-rtx6000.job',     0, 2),
        ('lpdq-4-2', 'quadro', 'run-lime-predict-p6-c-f-42-rtx6000.job',     0, 2),
        ('lpdq-4-3', 'quadro', 'run-lime-predict-p6-c-f-43-rtx6000.job',     0, 2),
    ]
    opt_max_submit_per_occasion = 1
    opt_project_dir = os.environ['PRJ_DIR']
    opt_stopfile = 'stop-fill-queue'
    opt_stop_check_interval = 12.0
    opt_submit_interval = 1800.0
    opt_probe_interval  = 300.0
    opt_username = getpass.getuser()
    start_time = time.time()
    earliest_next_submit = start_time
    has_tasks = {}
    while True:
        exit_reason = None
        if opt_stopfile and os.path.exists(opt_stopfile):
            exit_reason = 'Found stop file'
        if exit_reason:
            print('\n*** %s ***\n' %exit_reason)
            sys.exit(0)
        now = time.time()
        if now < earliest_next_submit:
            # wait no more than `opt_stop_check_interval` seconds so that
            # stop conditions are checked regularly and avoid leaving a
            # very short waiting period to the end
            wait = min(2*opt_stop_check_interval, earliest_next_submit - now)
            if wait > opt_stop_check_interval:
                wait = wait / 2
            time.sleep(wait)
            continue
        # get queue state
        command = ['squeue', '--noheader', '--user=' + opt_username]
        try:
            output = subprocess.check_output(command) # an exception is raised if the command fails
        except subprocess.CalledProcessError:
            print('Error checking job queue, trying again in a minute')
            earliest_next_submit = time.time() + 60.0
            continue
        queue = defaultdict(lambda: 0)
        for row in output.split('\n'):
            row.rstrip()
            if not row:
                continue
            fields = row.split()
            # light check for expected format
            assert len(fields) >= 8
            assert fields[3] == opt_username
            # extract data
            job_name = fields[2]
            job_name_prefix = job_name.split('-')[0]
            inbox = job_name[-3:]
            job_state = fields[4]
            queue[(job_name_prefix, job_state)] += 1
            queue[(job_name, job_state)] += 1
            queue[(inbox, job_state)] += 1
            resource = get_resource(opt_jobs, job_name_prefix)
            queue[(resource, job_state)] += 1
        print('My jobs at', time.ctime(now))
        for key in sorted(list(queue.keys())):
            print('\t%r with frequency %d' %(key, queue[key]))
        # check what may be needed
        non_empty_inboxes = 0
        for inbox in '1-1 1-2 1-3 2-1 2-2 2-3 3-1 3-2 3-3 4-1 4-2 4-3'.split():
            inbox_path = '%s/c-f-%s/tasks' %(opt_project_dir, inbox)
            has_tasks[inbox] = False
            entries = os.listdir(inbox_path)
            for inbox_f in entries:
                if inbox_f.endswith('.new'):
                    has_tasks[inbox] = len(entries) # upper bound for # tasks (there can be other entries)
                    non_empty_inboxes += 1
                    break
            if non_empty_inbox >= random.randrange(3,7):
                break
        # check what to submit
        std_jobs = []
        prio_jobs = []
        secondary_prio_jobs = []
        for job_item in opt_jobs:
            std_jobs.append(job_item)
        random.shuffle(prio_jobs)
        random.shuffle(secondary_prio_jobs)
        random.shuffle(std_jobs)
        opt_jobs = prio_jobs + secondary_prio_jobs + std_jobs
        n_submitted = 0
        for job_name, resource, script_name, max_waiting, max_running in opt_jobs:
            inbox = job_name[-3:]
            job_name_prefix = job_name.split('-')[0]
            if not has_tasks[inbox]:
                continue
            if queue[(resource, 'PD')] > max_waiting:
                continue
            if queue[(job_name_prefix, 'PD')] > max_waiting:
                continue
            if queue[(job_name_prefix, 'R')] > max_running:
                continue
            # submit job
            sys.stdout.flush()   # make sure command output appears after our last output
            #command = ['sbatch', '/'.join((opt_script_dir, script_name))]
            command = ['sbatch', script_name]
            try:
                subprocess.call(command)
            except subprocess.CalledProcessError:
                print('Error submitting job, trying again in a minute')
                earliest_next_submit = time.time() + 60.0
                break
            print('Submitted %s (%s) with %d entries(s) in inbox' %(
                job_name, script_name, has_tasks[inbox]
            ))
            # move forward time for next job submission
            now = time.time()
            while earliest_next_submit <= now:
                earliest_next_submit += opt_submit_interval
            # limit how many jobs to submit at each occasion
            n_submitted += 1
            if n_submitted >= opt_max_submit_per_occasion:
                break
        now = time.time()
        while earliest_next_submit <= now:
            earliest_next_submit += opt_probe_interval
        sys.stdout.flush()

if __name__ == "__main__":
    main()


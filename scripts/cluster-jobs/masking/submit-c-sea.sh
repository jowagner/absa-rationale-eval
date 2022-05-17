#!/usr/bin/env bash

for T in `cat c-sea.tasks` ; do
    echo "getting ready to submit job..."
    for I in 5 4 3 2 1 ; do
        echo "$I"
        sleep 1
    done
    echo "submitting ${T}..."
    sbatch $T
    echo "submitted $T"
    sleep 1495
done


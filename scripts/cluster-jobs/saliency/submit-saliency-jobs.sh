#!/usr/bin/env bash

for T in `cat saliency.tasks` ; do
    echo
    echo "getting ready to submit job..."
    for I in 9 8 7 6 5 4 3 2 1 ; do
        echo "$I"
        sleep 1
    done
    date
    echo "submitting ${T}..."
    sbatch $T
    echo "submitted $T"
    date
    sleep 18000
    echo
done


#!/usr/bin/env bash

for FOLDER in c-f-?-? ; do 
    echo $FOLDER
    fgrep "== Final summary for ('-', 'q0', 'test', None) ==" -A 2050 $FOLDER/saliency-morewio-xfw-stdout.txt > $FOLDER/final-test-summary.txt
done

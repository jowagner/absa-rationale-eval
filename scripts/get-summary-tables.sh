#!/usr/bin/env bash

for FOLDER in c-f-?-? ; do 
    fgrep "== Final summary for ('-', 'q0', 'test', None) ==" -A 2050 $FOLDER/saliency-morewio-xfw-stdout.txt  > $FOLDER/final-test-summary-for-R-IG.txt
    fgrep "== Final summary for ('-', 'q0', 'test', None) ==" -A 2050 $FOLDER/saliency-onepoint-xfw-stdout.txt > $FOLDER/final-test-summary-for-R-PG.txt
    for LIMECODE in M N S X ; do
        echo $FOLDER $GRADTYPE
        fgrep "== Final summary for ('" -A 2050 $FOLDER/lime-${LIMECODE}-fscores-te.txt > $FOLDER/final-test-summary-for-R-LIME-${LIMECODE}.txt
    done
done

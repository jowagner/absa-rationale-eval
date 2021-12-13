#!/bin/bash

for L in L25 L50 L75; do

    tail -n 5 c-f*/stdout-training-with-${L}-aio.txt | scripts/masking-summary-to-tsv.py Full     R-$L >  fig2-acc-masking-rationales-comb-${L}.tsv
    tail -n 5 c-s*/stdout-training-with-${L}-aio.txt | scripts/masking-summary-to-tsv.py R-$L     R-$L >> fig2-acc-masking-rationales-comb-${L}.tsv
    tail -n 5 c-o*/stdout-training-with-${L}-aio.txt | scripts/masking-summary-to-tsv.py Other-$L R-$L >> fig2-acc-masking-rationales-comb-${L}.tsv
    tail -n 5 c-a*/stdout-training-with-${L}-aio.txt | scripts/masking-summary-to-tsv.py All-$L   R-$L >> fig2-acc-masking-rationales-comb-${L}.tsv

done

#!/bin/bash

tail -n 5 c-[fnsoa]-?-?/stdout-training-with-*-aio.txt | scripts/masking-summary-to-tsv.py > table-3-acc-masking-rationales-comb.tsv
 

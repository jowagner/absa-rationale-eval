#!/bin/bash

#tail -n 6 c-[fnsoa]-?-?/stdout-training-with-*-aio.txt | scripts/masking-summary-to-tsv.py --domains > table-3-acc-masking-rationales-comb.tsv
tail -n 6 c-[fnsoa]-?-?/stdout-training-with-*-aio.txt | scripts/masking-summary-to-tsv.py  > table-3-acc-masking-rationales-comb.tsv
 

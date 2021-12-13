#!/bin/bash

echo "Micro average f-score:"
grep -A 5 -E "Final summary.*length oracle.*test" c-f-*/saliency-length-oracle-stdout.txt | grep -E "[.][0-9]" | cut -f9 | sort -n

echo "Inversely-weighted average f-score:"
grep -A 5 -E "Final summary.*length oracle.*test" c-f-*/saliency-length-oracle-stdout.txt | grep -E "[.][0-9]" | cut -f21 | sort -n

echo "Macro average f-score:"
grep -A 5 -E "Final summary.*length oracle.*test" c-f-*/saliency-length-oracle-stdout.txt | grep -E "[.][0-9]" | cut -f13 | sort -n



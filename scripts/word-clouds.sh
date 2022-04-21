#!/bin/bash

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# assumes the commented out line in scripts/extract-aio.sh to
# produce .tmp files with intermediate steps are included in
# the appropriate place

# requirement: https://github.com/amueller/word_cloud

cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tO\tO$' | cut -f1 | LC_ALL=C sort | uniq -c | sort -n -r > top-words-for-r-is-o-and-se-is-o.tsv
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tO\tI$' | cut -f1 | LC_ALL=C sort | uniq -c | sort -n -r > top-words-for-r-is-o-and-se-is-i.tsv
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tI\tI$' | cut -f1 | LC_ALL=C sort | uniq -c | sort -n -r > top-words-for-r-is-i-and-se-is-i.tsv
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tI\tO$' | cut -f1 | LC_ALL=C sort | uniq -c | sort -n -r > top-words-for-r-is-i-and-se-is-o.tsv

cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tI\tO$' | cut -f1 > top-text-for-r-is-i-and-se-is-o.txt
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tI\tI$' | cut -f1 > top-text-for-r-is-i-and-se-is-i.txt
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tO\tI$' | cut -f1 > top-text-for-r-is-o-and-se-is-i.txt
cat c-f*/t*-*-L50-step1.tmp | cut -f5,6,7 | fgrep $'\t' | grep -E $'\tO\tO$' | cut -f1 > top-text-for-r-is-o-and-se-is-o.txt

for R in i o ; do
    for S in i o ; do 

        wordcloud_cli --background white --color black      \
            --width 1280 --height 960                        \
            --text top-text-for-r-is-${R}-and-se-is-${S}.txt  \
            --no_collocations --no_normalize_plurals           \
            --include_numbers                                   \
	    --stopwords scripts/stopword-distractors.txt         \
            --imagefile wordcloud-for-r-is-${R}-and-se-is-${S}.png

    done
done


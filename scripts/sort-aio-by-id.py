#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# restore item order in .aio file augmented with review ID, sentence index
# and opinion index provided in first input column

import sys

example = """
1:0:0     All     I
1:0:0     I       O
1:0:0     can     I
1:0:0     say     I
1:0:0     is      O
1:0:0     W-O-W   I
1:0:0     .       O

1:2:0     It      O
"""

def sort_key(sent_id):
    fields = sent_id.split(':')
    assert len(fields) == 3     # (review_id, sentence_index, opinion_index)
    return (fields[0], int(fields[1]), int(fields[2]))

sentences = []
sentence = []
current_id = None
while True:
    line = sys.stdin.readline()
    if not line or line.isspace():
        if sentence:
            sentences.append((sort_key(current_id), sentence))
            sentence = []
            current_id = None
        if not line:
            break
        continue
    fields = line.split('\t')
    if current_id:
        assert current_id == fields[0]
    else:
        current_id = fields[0]
    sentence.append(line)

sentences.sort()
for _, sentence in sentences:
    for line in sentence:
        sys.stdout.write(line)
    sys.stdout.write('\n')


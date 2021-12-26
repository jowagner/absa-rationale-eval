#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

odd = even = 0

while True:
    line = sys.stdin.readline()
    if not line:
        break
    n = int(line)
    if n % 2:
        odd += 1
    else:
        even += 1

sys.stdout.write('%d odd, %d even\n' %(odd, even))

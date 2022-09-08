#!/usr/bin/env python
# -*- coding: ascii -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

def usage():
    print('Usage: cat t*-*-M50-wcloud.tsv | $0 [options]')
    # TODO: print more details how to use this script

    # note P50-wcloud.tsv still uses L50 prefix

selected_aio = 'M50'
selected_set = 'training'
n_examples   = 15
treat_dev_as_training = True
unshuffled   = True
seed         = None
debug        = False

while len(sys.argv) > 1 and sys.argv[1][:2] in ('--', '-h', '-n'):
    option = sys.argv[1].replace('_', '-')
    del sys.argv[1]
    if option in ('-h', '--help'):
        usage()
        sys.exit(0)
    elif option == '--debug':
        debug = True
    elif option == '--exclude-dev-from-training':
        treat_dev_as_training = False
    elif option in ('--shuffle', '--randomise-order'):
        unshuffled = False
    elif option in ('--set', '--set-type'):
        selected_set = sys.argv[1]
        if selected_set.lower() in ('any', 'both', 'none'):
            selected_set = None
        del sys.argv[1]
    elif option == '--aio':
        selected_aio = sys.argv[1]
        del sys.argv[1]
    elif option in ('-n', '--num-examples', '--number-of-examples'):
        n_examples = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--seed', '--random-seed'):
        seed = sys.argv[1]
        unshuffled = False   # --seed implies --shuffle
        del sys.argv[1]
        if seed == '0':
            # 0 = use system default
            seed = None
            if debug: sys.stderr.write('PRNG using system seed\n')
    else:
        print('Unknown option', option)
        usage()
        sys.exit(1)

if len(sys.argv) > 1:
    usage()
    sys.exit(1)

example_input = """
L50             laptop          training        1:0:0           All             I               O
L50             laptop          training        1:0:0           I               O               O
L50             laptop          training        1:0:0           can             I               O
L50             laptop          training        1:0:0           say             I               O
L50             laptop          training        1:0:0           is              O               O
L50             laptop          training        1:0:0           W-O-W           I               I
L50             laptop          training        1:0:0           .               O               O

--
L50             laptop          training        1:2:0           It              O               O
"""

def sort_key(sent_id):
    fields = sent_id.split(':')
    assert len(fields) == 3     # (review_id, sentence_index, opinion_index)
    return (fields[0], int(fields[1]), int(fields[2]))

domain2info = {}

def add_sentence(sentence, current_id, domain):
    global domain2info
    review_id, sentence_index, opinion_index = current_id.split(':')
    sentence_index = int(sentence_index)
    opinion_index = int(opinion_index)
    if not domain in domain2info:
        domain2info[domain] = {}
    sid2items = domain2info[domain]
    sid = (review_id, sentence_index)
    if not sid in sid2items:
        sid2items[sid] = []
    sid2items[sid].append((opinion_index, sentence))

sentence = []
current_id = None
current_domain = None
while True:
    line = sys.stdin.readline()
    if not line or line.isspace():
        if sentence:
            add_sentence(sentence, current_id, current_domain)
            sentence = []
            current_id = None
            current_domain = None
        if not line:
            break
        continue
    fields = line.rstrip().split('\t')
    if fields[0] != selected_aio:
        continue
    if selected_set:
        current_set = fields[2]
        if treat_dev_as_training and current_set == 'dev':
            current_set = 'training'
        if current_set != selected_set:
            continue
    if current_id:
        assert current_id == fields[3]
        assert current_domain == fields[1]
    else:
        current_id = fields[3]
        current_domain = fields[1]
    sentence.append(fields[-3:])

def latex_text(s):
    s = s.replace('%', '\\%')
    s = s.replace('"', "''")
    s = s.replace('$', '\\$')
    return s

for domain in sorted(list(domain2info.keys())):
    sys.stdout.write('\\subsubsection{%s Domain} %% %s\n' %(
        domain.title(), selected_aio
    ))
    sid2items = domain2info[domain]
    sid_keys = list(sid2items.keys())
    if debug: sys.stderr.write('%d sentence IDs for %s domain\n' %(len(sid_keys), domain))
    if seed or unshuffled:
        sid_keys.sort()
    if seed:
        domain_seed = '%s:%s' %(domain, seed)
        import hashlib
        if type(b'') is not str:
            # Python 3
            domain_seed = domain_seed.encode('UTF-8')
        # convert string to int consistently across Python versions
        if debug: sys.stderr.write('Hashing string %r with %d bytes derived from (%s, %s)\n' %(domain_seed, len(domain_seed), domain, seed))
        numeric_seed = int(hashlib.sha512(domain_seed).hexdigest(), 16)
        if debug: sys.stderr.write('Domain and seed hashed to %d\n' %numeric_seed)
        random.seed(numeric_seed)
    if not unshuffled:
        if debug:
            rng_state = random.getstate()
            rnd = [random.random(), random.random(), random.random()]
            test_seq = [1,2,3,4,5,6,7,8]
            random.shuffle(test_seq)
            rnd.append(test_seq)
            rnd = tuple(rnd)
            sys.stderr.write('%.3f %.3f %.3f %r\n' %rnd)
            random.setstate(rng_state)
        random.shuffle(sid_keys)
    examples_printed = 0
    is_first = True
    for sid_key in sid_keys:
        if examples_printed >= n_examples:
            break
        sentences = sid2items[sid_key]
        sentences.sort()
        for opinion_index, sentence in sentences:
            sys.stdout.write('\n%% Sentence %r, item %d\n' %(sid_key, opinion_index + 1))
            if not is_first:
                sys.stdout.write('\\vspace{0.3cm}\n')
            is_first = False
            sys.stdout.write('\\noindent\n')
            inside_se = False
            inside_r  = False
            needs_space = False
            for fields in sentence:
                token = fields[0]
                token_in_se = fields[2] is 'I'
                token_in_r  = fields[1] is 'I'
                if inside_se == token_in_se and inside_r == token_in_r:
                    # no transition / just adding a token
                    if needs_space:
                        sys.stdout.write(' ')
                    sys.stdout.write(latex_text(token))
                    needs_space = True
                else:
                    # state transition
                    if token_in_se and not inside_se:
                        # entering an SE
                        if inside_r:
                            # ensure proper nesting of { }
                            sys.stdout.write('} ')
                        elif needs_space:
                            sys.stdout.write(' ')
                        sys.stdout.write('\\textbf{')
                        inside_se = True
                        if token_in_r:
                            sys.stdout.write('\\underline{')
                        inside_r = token_in_r
                    elif inside_se and not token_in_se:
                        # leaving an SE
                        if inside_r:
                            # ensure proper nesting of { }
                            sys.stdout.write('}')
                        sys.stdout.write('} ')
                        inside_se = False
                        if token_in_r:
                            sys.stdout.write('\\underline{')
                        inside_r = token_in_r
                    elif token_in_r and not inside_r:
                        # entering rationale
                        if needs_space:
                            sys.stdout.write(' ')
                        sys.stdout.write('\\underline{')
                        inside_r = True
                    elif inside_r and not token_in_r:
                        # leaving rationale
                        sys.stdout.write('} ')
                        inside_r = False
                    sys.stdout.write(latex_text(token))
                    needs_space = True
            if inside_r:
                sys.stdout.write('}')
            if inside_se:
                sys.stdout.write('}')
            sys.stdout.write('\n')
            examples_printed += 1

# ABSA Rationale Evaluation

We evaluate rationales in the task of aspect-based sentiment analysis (ABSA) against manually annotated sentiment expressions.
The software in this repository
* fine-tunes English BERT-Base in the SemEval 2016 ABSA shared task 5, subtask "Phase B"
* applies methods for identifying a subset of the input tokens that is most influential on the sentiment prediction (a salience map, rationale or explanation)
* evaluates the rationales against manually annotated sentiment expressions relevant for the selected aspect term


## Data

We use the sentiment expression annotation of
[Kaljahi, Jennifer Foster (2018)](https://aclanthology.org/W18-6222/)
available from
https://opengogs.adaptcentre.ie/rszk/sea
together with the
[SemEval 2016 task 5](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
subtask 1 data.


## Dependencies

| Dependency   | Development Machine | Experiment (Cluster) |
| ------------ | ------------------- | -------------------- |
| CUDA + CUDNN | 11.1                | 10.2 |
| PyTorch      | 1.9.0               | 1.9.0 |
| Hugging Face transformers | 4.8.1  | 4.9.1 |
| pytorch-lightning | 1.3.7.post0    | 1.4.1 |
| Python       | 3.6.13 - 3.6.15     | 3.7.3 |
| wordcloud    |                     | n/a   |
| imbalanced-learn |                 |       |

* `create-venv-pytorch*.sh`: how the environment was set up
* `pip-freeze-snapshot*.txt`: all version numbers, including for indirect dependencies


## Training Classifiers

Train basic classifers "Full", "None", "SE" and "not SE":
1. `scripts/cluster-jobs/masking/gen-training-jobs.py`
2. Submit `run-train-c-sea-[fson]-[1-3]1-to-[1-3]3.job` (12 jobs, each training 3 classifiers)

First 2 rows of Table 1 (and some of the appendix tables) are ready:
 - Run `scripts/get-accuracies-for-masking-rationales.sh` and inspect output file `results-masking-diagonal.tex`

Get saliency maps and detailed log files:
1. Run `scripts/cluster-jobs/saliency/gen-saliency-jobs.py`
2. Submit `run-saliency-c-f-??.job` (9 jobs, CPU-only)

Train R@ classifiers:
1. Run `scripts/extract-aio.sh` to get `.aio` files for rationales with length .25, .50 and .75
2. Submit `run-train-c-L[257][05]-[so]-[1-3]1-to-[1-3]3.job` (18 jobs, each training 3 classifiers)

Train U-SE and not-U-SE classifiers:
 - Submit `run-train-c-union-[so]-[1-3]1-to-[1-3]3.job` (6 jobs, each training 3 classifiers; `.aio` files are created inside each job)

Train A@ classifiers:
 - Submit `run-train-c-RND[257][05]-[os]-?1-to-?3.job` (18 jobs, each training 3 classifiers, `.aio` files are created inside each job)

Generate Table 1 (appendix tables are also partially ready):
 - Run `scripts/get-accuracies-for-masking-rationales.sh` and inspect output file `results-masking-diagonal.tex`

Training jobs not submitted above are only needed for results reported in the appendix of our paper.
The `f' (Full) and `n' (None) classifiers should not vary across the different settings (sea, union, R25, etc.).

It should be possible to train the U-SE and A@ classifiers in parallel with the first step but we did not test this.

## Agreement with Sentiment Expressions

To produce Figure 1:
1. Run `scripts/get-summary-tables.sh` to prepare `final-test-summary.txt` in each `c-f-?-?` folder
2. Run `scripts/summary-table-to-graph-data.py` to produce 6 `.tsv` files
3. Use spreadsheet application to create graph from `FL-inv-weighted-micro.tsv` (FL = F-score over length)

For the above steps, saliency maps are only needed for test data.
To try different saliency methods, `train-classifier.py` can be run
with `--dev-and-test-saliencies-only`, speeding up the process about 4x.


## Length Oracle

SE agreement statistics for rationales with gold length are included in the `saliency-onepoint-xfw-stdout.txt` log files for gradient-based rationales and `lime-M-fscores-te.txt for LIME-based rationales.

Run
```
for I in c-s-[1234]-? ; do scripts/get-se-length-distribution.py --aio-prefix $I/local-aio-M50/ test | fgrep "Overall span counts" -A 999 > $I/span-counts-M50.txt ; done
for I in c-s-[1234]-? ; do scripts/get-se-length-distribution.py --aio-prefix $I/local-aio-P50/ test | fgrep "Overall span counts" -A 999 > $I/span-counts-P50.txt ; done
paste c-s-*/span-counts-P50.txt
paste c-s-*/span-counts-M50.txt
```
to obtain span counts for all 12 runs. Remove columns 3, 5, 7, etc. in a spreadsheet, get row totals (without column 1) and get percentages.


## Sentence Length Distribution

Run `scripts/get-se-length-distribution.py --aio-prefix data/ test`
and add the `rlen` vectors for each domain to obtain overall counts for Figure 4.


## Confusion Matrices

Run `get-confusion-matrices.py`.
We used the output to find out that 3.1% of test instances can be predicted correctly
as neutral by the SE models simply based on not the fact that these many test instances have zero SE tokens and
are neutral.


## Word Cloud

1. Check that the c-f-?-? folders contain `-wcloud.tsv` files for the desired setting. Modify and re-run `extract-aio.sh` as needed. Default is to produce word clouds for R@.5 only.
2. Install https://github.com/amueller/word_cloud e.g. with `pip install wordcloud`
3. Run `word-clouds.sh P` to produce word clouds for gradient-based rationales
3. Run `word-clouds.sh M` to produce word clouds for LIME-based rationales

Appendix tables for word clouds:
`gen-word-count-tables.py` expects to be run in a folder containing 4 files with the pattern `top-words-for-r-is-[io]-and-se-is-[io].tsv`.
These should be in the `wc-P` and `wc-M` folder of the word cloud files.



## Get Examples with Sentiment Expressions and Rationales

```
cat c-f-1-1/t*-*-P50-wcloud.tsv | scripts/get-examples.py --seed 101 --aio L50  > examples-grad-shuffled.tex
cat c-f-1-1/t*-*-M50-wcloud.tsv | scripts/get-examples.py --seed 101 --aio M50  > examples-lime-shuffled.tex
cat c-f-1-1/t*-*-P50-wcloud.tsv | scripts/get-examples.py --aio L50  > examples-grad-ordered.tex
cat c-f-1-1/t*-*-M50-wcloud.tsv | scripts/get-examples.py --aio M50  > examples-lime-ordered.tex
```

Note that the shuffling differs between Python 2 and 3 (starting with version 3.2 according to
https://stackoverflow.com/questions/38943038/difference-between-python-2-and-3-for-shuffle-with-a-given-seed).
We use Python 3 to ease reproduction of results.


## Citation

If you use this code for your research, please cite our paper:

Joachim Wagner and Jennifer Foster (to appear at ACL 2023) 
Investigating the Saliency of Sentiment Expressions in Aspect-Based Sentiment Analysis.
In *Findings of the Association for Computational Linguistics: ACL 2023*,
Toronto, Canada, 09-14 July 2023,
Association for Computational Linguistics.


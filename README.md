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


## Agreement with Sentiment Expressions

To produce Figure 1:
1. Run `scripts/get-summary-tables.sh` to prepare `final-test-summary.txt` in each `c-f-?-?` folder
2. Run `scripts/summary-table-to-graph-data.py` to produce 6 `.tsv` files
3. Use spreadsheet application to create graph from `FL-inv-weighted-micro.tsv` (FL = F-score over length)


## Length Oracle

1. SE agreement statistics for rationales with gold length are included in the `saliency-morewio-xfw-stdout.txt` log files.
2. `get-length-oracle-scores.sh`


## Sentence Length Distribution

`get-se-length-distribution.py`
for Figure 2


## Confusion Matrices

`get-confusion-matrices.py`
We used the output to find out that 3.1% of test instances can be predicted correctly
as neutral by the SE models simply based on not the fact that these many test instances have zero SE tokens and
are neutral.


## Word Cloud

`word-clouds.shword-clouds.sh`
requires https://github.com/amueller/word_cloud that can be installed with
`pip install wordcloud`


## Get Examples with Sentiment Expressions and Rationales

`get-examples.py`


## Citation

Here we will provide details on the paper this software is first used in when it is accepted.


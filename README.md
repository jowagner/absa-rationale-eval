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
2. Run `run-train-c-sea-[fson]-[1-3]1-to-[1-3]3.job` (12 jobs) to get "Full", "SE", "not SE" and "None" models

First 2 rows of Table 1 (and some of the appendix tables) are ready:
3. Run `scripts/get-accuracies-for-masking-rationales.sh` and inspect output file `results-masking-diagonal.tex`

Get saliency maps and detailed log files:
4. Run `scripts/cluster-jobs/saliency/gen-saliency-jobs.py`


TODO:
 - Train U-SE and not U-SE classifiers
 - `extract-aio.sh` to get .aio files for our rationales R@.25, R@.50 and R@.75 (L25, L50, L75 aio)
 - Train R@ classifiers
 - get .aio files for random saliency maps, e.g. A50 should contain all tokens of A25
 - Train A@ classifiers
 - What is needed to create Fig 1?


## Citation

Here we will provide details on the paper this software is first used in when it is accepted.


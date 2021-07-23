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

We develop this software with CUDA 11.1 + CUDNN, PyTorch 1.9.0,
Hugging Face transformers and pytorch-lightning in Python 3.6.13.
* `create-venv-pytorch.sh`: how the environment was set up
* `pip-freeze-snapshot.txt`: all version numbers, including for indirect dependencies


## Citation

Here we will provide details on the paper this software is first used in when it is accepted.


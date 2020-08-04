# STC-MAML-PyTorch
## A PyTorch Implementation of an Extended MAML Algorithm for Few-Shot Spoken Term Classification
This repository presents an extended-MAML approach to address the user-defined spoken term classification task. For a detailed description of the work, please refer to [our paper](https://arxiv.org/abs/1812.10233)。

Our main contributions are:
+ We investigate the performance of MAML, as one of the most popular few-shot learning solutions, on a speech task.
+ We extend the original MAML to solve a more realistic N+M-way, K-shot problem.
+ We investigate how much a user-defined spoken term classification system can get close to a predefined one.
 
Our implementation is based on a PyTorch implementation of MAML [MAML-PyTorch](https://github.com/dragen1860/MAML-Pytorch).

## Prerequisites
+ python: 3.x
+ PyTorch: 0.4+

## Dataset - Google Speech Commands dataset v2
1. We use the raw data from the dataset which contains 35 keywords: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'cat', 'tree', 'house', 'bird', 'visual', 'backward', 'follow', 'forward', 'learn', 'sheila', 'bed', 'dog', 'happy', 'marvin', 'wow'.
2. We split the dataset into different parts to satisfy our experimental setting:
```shell
data/
├──speech_commands/
    ├──yes
        ├──00f0204f_nohash_0.wav
        ├──d962e5ac_nohash_1.wav
        ...
├──train_commands.csv
├──test_commands.csv
├──train_digits.csv
├──test_digits.csv
├──unknown_train.csv
├──unknown_test.csv

```
## Train and test
1. Run `python train.py` followed by a series of arguments:
```shell
--task_type
--k_spt_train
--k_qry_train
--k_spt_unk_train
--k_qry_unk_train
--k_spt_silence_train
--k_qry_silence_train
--k_spt_test
--k_qry_test
--k_spt_unk_test
...
```
## Cite our paper
If the code and the work is useful to you, please cite it:
```shell
@article{chen2018investigation,
  title={An Investigation of Few-Shot Learning in Spoken Term Classification},
  author={Chen, Yangbin and Ko, Tom and Shang, Lifeng and Chen, Xiao and Jiang, Xin and Li, Qing},
  journal={arXiv preprint arXiv:1812.10233},
  year={2018}
}
```

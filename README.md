# STC-MAML-PyTorch
## A PyTorch Implementation of an Extended MAML Algorithm for Few-Shot Spoken Term Classification
This repository presents an extended-MAML approach to address the user-defined spoken term classification task. For a detailed description of the work, please refer to [our paper](https://arxiv.org/abs/1812.10233)。

Our main contributions are:
+ We investigate the performance of MAML, as one of the most popular few-shot learning solutions, on a speech task.
+ We extend the original MAML to solve a more realistic N+M-way, K-shot problem.
+ We investigate how much a user-defined spoken term classification system can get close to a predefined one.
 
The code is based on an PyTorch implementation of MAML [MAML-PyTorch](https://github.com/dragen1860/MAML-Pytorch).

## Prerequisites
+ python: 3.x
+ PyTorch: 0.4+

## Dataset - Google Speech Commands dataset v2
1. We use the raw data from the dataset which contains 35 keywords: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'cat', 'tree', 'house', 'bird', 'visual', 'backward', 'follow', 'forward', 'learn', 'sheila', 'bed', 'dog', 'happy', 'marvin', 'wow'.
2. We split the dataset into different parts to satisfy our experimental setting:

![filepaths](/img/filepaths.png =200x)

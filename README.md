# nn-from-scratch
Basic neural network from scratch (just Python and NumPy)

## Overview
This project contains my slightly modified interpretation of the code presented in Tariq Rashid's book [Make Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608).
In short, it is a 3-layer neural network built specifically for training and querying hand-written digits sourced from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Setup
If you are unfamiliar with [virtual environments](https://docs.python.org/3/library/venv.html), I suggest taking a brief moment to learn about them and set one up.
The Python docs provide a great [tutorial](https://docs.python.org/3/tutorial/venv.html) for getting started with virtual environments and packages.

This project's dependencies can be installed via pip:
```
pip install -r requirements.txt
```

## Usage
First, fetch a preprocessed version of the MNIST dataset using `fetch.py`:
```
python3 fetch.py
```

Then, train and query the dataset using `nn.py`:
```
python3 nn.py
```

Once the training is complete, the script will query all of the MNIST test images and compute an overall accuracy rating.

For better accuracy, try the following tweaks:
```
python3 nn.py --epochs 5 --hidden-nodes 200 --learning-rate 0.1
```

## Tweaks
Despite being quite specific, this neural network has a few parameters that can be used to adjust its behavior (and therefore its performance):
```
usage: nn.py [-h] [--epochs EPOCHS] [--hidden-nodes HIDDEN_NODES]
             [--learning-rate LEARNING_RATE] [--train TRAIN] [--query QUERY]

train and query the MNIST dataset

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training iterations
  --hidden-nodes HIDDEN_NODES
                        number of hidden nodes
  --learning-rate LEARNING_RATE
                        network learning rate
  --train TRAIN         dataset to use for training
  --query QUERY         dataset to use for querying
```

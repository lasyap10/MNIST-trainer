#!/bin/bash

mkdir -p data/mnist

curl -o data/mnist/train-images.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o data/mnist/train-labels.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o data/mnist/test-images.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o data/mnist/test-labels.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip data/mnist/train-images.gz
gunzip data/mnist/train-labels.gz
gunzip data/mnist/test-images.gz
gunzip data/mnist/test-labels.gz
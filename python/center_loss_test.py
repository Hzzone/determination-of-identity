# -*- coding: utf-8 -*-
import numpy as np
import os
# Make sure that caffe is on the python path:
# caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import random
import matplotlib.pyplot as plt
import pylab
import itertools
import distance



MODEL_FILE = '../mnist_center_loss/mnist_deploy.prototxt'
# decrease if you want to preview during training
PRETRAINED_FILE = '../mnist_center_loss/mnist_train_iter_10000.caffemodel'
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
TEST_DATA_FILE = os.path.join(caffe_root, 'data/mnist/t10k-images-idx3-ubyte')
TEST_LABEL_FILE = os.path.join(caffe_root, 'data/mnist/t10k-labels-idx1-ubyte')
n = 10000

with open(TEST_DATA_FILE, 'rb') as f:
    f.read(16) # skip the header
    raw_data = np.fromstring(f.read(n * 28*28), dtype=np.uint8)

with open(TEST_LABEL_FILE, 'rb') as f:
    f.read(8) # skip the header
    labels = np.fromstring(f.read(n), dtype=np.uint8)

print raw_data.shape
# reshape and preprocess
caffe_in = raw_data.reshape(n, 1, 28, 28) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
print caffe_in.shape
out = net.forward_all(data=caffe_in)

# test dataset output n*(x, y)
feat = out['ip1']
# feat = out['ip2']
print len(feat)
feat_combination = list(itertools.combinations(feat, 2))
labels_combination = list(itertools.combinations(labels, 2))

_same = []
_diff = []
for index, f in enumerate(feat_combination):
    if labels_combination[index][0] == labels_combination[index][1]:
        _same.append(f)
    else:
        _diff.append(f)
print len(_same)
totals = 6000
print len(_same)
print len(_diff)
_same = random.sample(_same, totals/2)
_diff = random.sample(_diff, totals/2)
print len(_same)
_same_distance = []
_diff_distance = []
correct = 0
for x in _same:
    _same_distance.append(distance.cosine_distnace(x[0], x[1]))
for x in _diff:
    _diff_distance.append(distance.cosine_distnace(x[0], x[1]))

threshold = 0
for x in range(totals/2):
    if _diff_distance[x] < threshold:
        correct = correct + 1
    if _same_distance[x] >= threshold:
        correct = correct + 1
print float(correct/totals)

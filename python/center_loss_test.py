# -*- coding: utf-8 -*-
import numpy as np
import os
# Make sure that caffe is on the python path:
# caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
caffe_root = '/home/bw/code/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
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
caffe.set_mode_gpu()
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

# reshape and preprocess
caffe_in = raw_data.reshape(n, 1, 28, 28) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
out = net.forward_all(data=caffe_in)

# test dataset output n*(x, y)
feat = out['ip1']
# feat = out['ip2']
print len(feat)
feat_combination = list(itertools.combinations(feat, 2))
labels_combination = list(itertools.combinations(labels, 2))
print len(feat_combination)
print len(labels_combination)

_same = []
_diff = []
for index, f in enumerate(feat_combination):
    if labels_combination[index][0] == labels_combination[index][1]:
        _same.append(f)
    else:
        _diff.append(f)
totals = 6000
_same = random.sample(_same, totals/2)
_diff = random.sample(_diff, totals/2)
_same_distance = []
_diff_distance = []
for x in _same:
    _same_distance.append(distance.cosine_distnace(x[0], x[1]))
for x in _diff:
    _diff_distance.append(distance.cosine_distnace(x[0], x[1]))

x_values = pylab.arange(-1.0, 1.01, 0.001)
y_values = []
for threshold in x_values:
    correct = 0
    for x in range(totals/2):
        if _diff_distance[x] < threshold:
            correct = correct + 1
        if _same_distance[x] >= threshold:
            correct = correct + 1
    y_values.append(float(correct)/totals)
max_index = np.argmax(y_values)
plt.title("threshold-accuracy curve")
plt.xlabel("threshold")
plt.ylabel("accuracy")
plt.plot(x_values, y_values)
plt.plot(x_values[max_index], y_values[max_index], '*', color='red', label="(%s, %s)"%(x_values[max_index], y_values[max_index]))
plt.legend()
plt.show()

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()

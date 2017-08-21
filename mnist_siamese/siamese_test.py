# -*- coding: utf-8 -*-
import numpy as np
import os
# Make sure that caffe is on the python path:
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
# caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import random
import matplotlib.pyplot as plt
import pylab

def cosine_distnace(v1, v2):
    cos = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos

def euclidean_distance(v1, v2):
    euc = np.sqrt(np.sum(np.square(v1 - v2)))
    return euc

def generate_accuracy_map(features, labels, totals=6000, threadhold=0):
    # the number of _diff and _same = totals/2
    _diff = []
    _same = []
    unique_labels = set(labels)
    length = len(unique_labels)
    diff_features = []
    for i in range(length):
        ith_features = features[labels==i]
        diff_features.append(ith_features)
        # 每个样本平均取
        for j in range(totals/(2*length)):
            x = random.randint(0, len(ith_features)-1)
            y = random.randint(0, len(ith_features)-1)
            first = ith_features[x]
            second = ith_features[y]
            # 这是所有相同的
            _same.append(cosine_distnace(first, second))
    # 这是不相同
    # 随机抽，不会抽在同一个类中
    for j in range(totals/2):
        while True:
            x = random.randint(0, length-1)
            y = random.randint(0, length-1)
            if x != y:
                break
        first = random.randint(0, len(diff_features[x])-1)
        second = random.randint(0, len(diff_features[y])-1)
        _diff.append(cosine_distnace(diff_features[x][first], diff_features[y][second]))
    correct = 0
    for elememt in _diff:
        if elememt < threadhold:
            correct = correct + 1
    for elememt in _same:
        if elememt >= threadhold:
            correct = correct + 1
    return float(correct)/totals


def plot_accuracy_map(features, labels, totals=6000):
    x_vaules = pylab.arange(-1.0, 1.0, 0.01)
    y_values = []
    for x in x_vaules:
        y_values.append(generate_accuracy_map(features=features, labels=labels, threadhold=x))
    max_index = np.argmax(y_values)
    print max_index
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_vaules, y_values)
    plt.plot(x_vaules[max_index], y_values[max_index], '.', label="(%s, %s)"%(x_vaules[max_index], y_values[max_index]))
    plt.legend()
    plt.show()


MODEL_FILE = './mnist_siamese.prototxt'
# decrease if you want to preview during training
PRETRAINED_FILE = './mnist_siamese_train_iter_50000.caffemodel'
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
feat = out['feat']

# 1 of test dataset
one = feat[labels==1]
# calculate euclidean distance
acc = euclidean_distance(one[0], one[1])
print acc

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

cos = cosine_distnace(one[0], one[1])
print cos


print generate_accuracy_map(features=feat, labels=labels)
plot_accuracy_map(feat, labels)

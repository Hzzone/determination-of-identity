import numpy as np
import os
# Make sure that caffe is on the python path:
caffe_root = '/home/bw/code/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import math

MODEL_FILE = './mnist_siamese.prototxt'
# decrease if you want to preview during training
PRETRAINED_FILE = './mnist_siamese_train_iter_50000.caffemodel'
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
feat = out['feat']

# 1 of test dataset
one = feat[labels==1]
# calculate euclidean distance
acc = np.sqrt(np.sum(np.square(one[0] - one[1])))
print acc

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


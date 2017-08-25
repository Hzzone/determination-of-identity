# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess

def ordinary_predict_two_sample(source, dimension=150, IMAGE_SIZE=227):
    pass

def siamese_predict_two_sample(source, dimension=150, IMAGE_SIZE=227):
    pass

def ordinary_predict_dataset(source, caffemodel, deploy_file,dimension=150, IMAGE_SIZE=227):
    pass

def ordinary_predict_dataset(source, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227):
    pass

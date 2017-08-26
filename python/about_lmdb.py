# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess
from itertools import combinations
import random

def generate_ordinary_lmdb(source, target="/Users/HZzone/Desktop/dete-data/train_lmdb", dimension=150, IMAGE_SIZE=227):
    env = lmdb.Environment(target, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for label, person in enumerate(os.listdir(source)):
            person_dir = os.path.join(source, person)
            one_person_samples = os.listdir(person_dir)
            for dicom_files in one_person_samples:
                s = os.path.join(person_dir, dicom_files)
                print s
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = dimension
                datum.height = IMAGE_SIZE
                datum.width = IMAGE_SIZE
                sample = preprocess.readManyDicom(s, IMAGE_SIZE, dimension)
                datum.data = sample.tobytes()
                datum.label = label
                str_id = "%s" % dicom_files
                txn.put(str_id, datum.SerializeToString())
            print "--------"

# source: the dataset folder
def generate_siamese_lmdb(source, target="/Users/HZzone/Desktop/dete-data/siamese_train_lmdb", dimension=150, IMAGE_SIZE=227):
    env = lmdb.Environment(target, map_size=int(1e12))
    dataset = generate_siamese_dataset(source)
    _same = dataset[0]
    _diff = dataset[1]
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 2*dimension
        datum.height = IMAGE_SIZE
        datum.width = IMAGE_SIZE
        sample = np.zeros((2*dimension, IMAGE_SIZE, IMAGE_SIZE))
        index = 0
        for same_sample in _same:
            label = 1
            sample[:dimension, :, :] = preprocess.readManyDicom(same_sample[0], IMAGE_SIZE, dimension)
            sample[dimension:, :, :] = preprocess.readManyDicom(same_sample[1], IMAGE_SIZE, dimension)
            datum.data = sample.tobytes()
            datum.label = label
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())
            index = index + 1
            print same_sample
            print "--------"
        print "***********"
        for diff_sample in _diff[:len(_same)]:
            label = 0
            sample[:dimension, :, :] = preprocess.readManyDicom(diff_sample[0], IMAGE_SIZE, dimension)
            sample[dimension:, :, :] = preprocess.readManyDicom(diff_sample[1], IMAGE_SIZE, dimension)
            datum.data = sample.tobytes()
            datum.label = label
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())
            index = index + 1
            print diff_sample
            print "--------"

# generate dataset path
# combines the samples, return _same and _diff
def generate_siamese_dataset(source):
    all_samples = []
    _same = []
    _diff = []
    for label, person in enumerate(os.listdir(source)):
        person_dir = os.path.join(source, person)
        one_person_samples = os.listdir(person_dir)
        for dicom_files in one_person_samples:
            sample = os.path.join(person_dir, dicom_files)
            all_samples.append(sample)
    sample_combinations = list(combinations(all_samples, 2))
    for one_comination in sample_combinations:
        if os.path.dirname(one_comination[0]) == os.path.dirname(one_comination[1]):
            _same.append(one_comination)
        else:
            _diff.append(one_comination)
    return _same, _diff

# not complete
def generate_hdf5(source, target="/Users/HZzone/Desktop/dete-data/train.h5", dimension=200, IMAGE_SIZE=227):
    for label, person in enumerate(os.listdir(source)):
        person_dir = os.path.join(source, person)
        one_person_samples = os.listdir(person_dir)
        for dicom_files in one_person_samples:
            s = os.path.join(person_dir, dicom_files)
            print s
            sample = np.zeros((dimension, IMAGE_SIZE, IMAGE_SIZE))
            datum = caffe.proto.caffe_pb2.Datum()
            for index, dicom_file in enumerate(os.listdir(s)):
                path = os.path.join(s, dicom_file)
                im = preprocess.preprocess(path, IMAGE_SIZE=IMAGE_SIZE)
                sample[index, :, :] = im
            datum.data = sample.tostring()
            datum.label = label
            str_id = "%s" % dicom_files
            txn.put(str_id, datum.SerializeToString())
        print "--------"
if __name__ == "__main__":
    # generate_ordinary_lmdb("/Users/HZzone/Desktop/mdzz", dimension=150)
    generate_siamese_lmdb("/Users/HZzone/Desktop/mdzz", dimension=150)
    # print generate_siamese_dataset("/Users/HZzone/Desktop/mdzz")

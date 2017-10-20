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
import distance
import statics

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
    with open("data.txt", "w") as f:
        f.write(str(_same)+'\n'+str(_diff))
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 2*dimension
        datum.height = IMAGE_SIZE
        datum.width = IMAGE_SIZE
        sample = np.zeros((2*dimension, IMAGE_SIZE, IMAGE_SIZE))
        # index = 0
        # for same_sample in _same:
        #     label = 1
        #     sample[:dimension, :, :] = preprocess.readManyDicom_sorted(same_sample[0], IMAGE_SIZE, dimension)
        #     sample[dimension:, :, :] = preprocess.readManyDicom_sorted(same_sample[1], IMAGE_SIZE, dimension)
        #     datum.data = sample.tobytes()
        #     datum.label = label
        #     str_id = "%8d" % index
        #     txn.put(str_id, datum.SerializeToString())
        #     index = index + 1
        #     print same_sample
        #     print "--------"
        # print "***********"
        # for diff_sample in _diff:
        #     label = 0
        #     sample[:dimension, :, :] = preprocess.readManyDicom_sorted(diff_sample[0], IMAGE_SIZE, dimension)
        #     sample[dimension:, :, :] = preprocess.readManyDicom_sorted(diff_sample[1], IMAGE_SIZE, dimension)
        #     datum.data = sample.tobytes()
        #     datum.label = label
        #     str_id = "%8d" % index
        #     txn.put(str_id, datum.SerializeToString())
        #     index = index + 1
        #     print diff_sample
        #     print "--------"
        _same = list(_same)
        _diff = list(_diff)
        all_samples = []
        for same_sample in _same:
            same_sample = list(same_sample)
            same_sample.append(1)
            all_samples.append(same_sample)
        for diff_sample in _diff:
            diff_sample = list(diff_sample)
            diff_sample.append(0)
            all_samples.append(diff_sample)

        # all_samples.extend(_same)
        # all_samples.extend(_diff)
        random.shuffle(all_samples)
        random.shuffle(all_samples)

        print all_samples
        print len(all_samples)

        for index, one_sample in enumerate(all_samples):
            print one_sample
            sample[:dimension, :, :] = preprocess.readManyDicom_sorted(one_sample[0], IMAGE_SIZE, dimension)
            sample[dimension:, :, :] = preprocess.readManyDicom_sorted(one_sample[1], IMAGE_SIZE, dimension)
            datum.data = sample.tobytes()
            datum.label = one_sample[2]
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())

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
    for one_combination in sample_combinations:
        if os.path.dirname(one_combination[0]) == os.path.dirname(one_combination[1]):
            _same.append(one_combination)
        else:
            _diff.append(one_combination)
    random.shuffle(_diff)
    return _same, _diff[:len(_same)]



    # return float(correct) / totals
if __name__ == "__main__":
    # generate_ordinary_lmdb("/Users/HZzone/Desktop/mdzz", dimension=150)
    statics.delete_DS_Store("/Users/HZzone/Desktop/determination-data/train")
    statics.delete_DS_Store("/Users/HZzone/Desktop/determination-data/test")
    generate_siamese_lmdb("/Users/HZzone/Desktop/determination-data/train", target="/Users/HZzone/Desktop/determination-data/siamese_train_lmdb", dimension=250)
    generate_siamese_lmdb("/Users/HZzone/Desktop/determination-data/test", target="/Users/HZzone/Desktop/determination-data/siamese_test_lmdb", dimension=250)
    # print generate_siamese_dataset("/Users/HZzone/Desktop/mdzz")

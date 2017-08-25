# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess

def generate_lmdb(source, target="/Users/HZzone/Desktop/dete-data/train_lmdb", dimension=200, IMAGE_SIZE=227):
    env = lmdb.Environment(target, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for label, person in enumerate(os.listdir(source)):
            person_dir = os.path.join(source, person)
            one_person_samples = os.listdir(person_dir)
            for dicom_files in one_person_samples:
                s = os.path.join(person_dir, dicom_files)
                print s
                sample = np.zeros((dimension, IMAGE_SIZE, IMAGE_SIZE))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = dimension
                datum.height = IMAGE_SIZE
                datum.width = IMAGE_SIZE
                for index, dicom_file in enumerate(os.listdir(s)):
                    path = os.path.join(s, dicom_file)
                    im = preprocess.preprocess(path, IMAGE_SIZE=IMAGE_SIZE)
                    sample[index, :, :] = im
                datum.data = sample.tobytes()
                datum.label = label
                str_id = "%s" % dicom_files
                txn.put(str_id, datum.SerializeToString())
            print "--------"


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
    generate_lmdb("/Users/HZzone/Desktop/mdzz")
    # generate_lmdb("/Users/HZzone/Desktop/mdzz/0001874926")

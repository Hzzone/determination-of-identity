# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
caffe_root = '/home/hzzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import preprocess
import distance
import about_lmdb
import pylab
import matplotlib.pyplot as plt
import dicom
import preparation

def ordinary_predict_two_sample(source1, source2, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    data = np.zeros((2, dimension, IMAGE_SIZE, IMAGE_SIZE))
    data[0, :, :, :] = preprocess.readManyDicom(source=source1, IMAGE_SIZE=IMAGE_SIZE, dimension=dimension)
    data[1, :, :, :] = preprocess.readManyDicom(source=source2, IMAGE_SIZE=IMAGE_SIZE, dimension=dimension)
    # only for test LeNet
    data = data * 0.00390625
    net.blobs['data'].data[...] = data
    output = net.forward()
    first_sample_feature = output[LAST_LAYER_NAME][0]
    second_sample_feature = output[LAST_LAYER_NAME][1]
    print distance.cosine_distnace(first_sample_feature, second_sample_feature)


# source: Test dataset
def ordinary_predict_dataset(source, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1", save_features_file="features.txt"):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    # patch = about_lmdb.generate_siamese_dataset(source)
    # samples = []
    # for root, dirs, files in os.walk(source):
    #     for dicom_file in files:
    #         samples.append(root)
    #         break
    person_samples = os.listdir(source)
    # one_person_samples = os.listdir(source)
    samples = {}
    for one_person_samples in person_samples:
        path = os.path.join(source, one_person_samples)
        temps = os.listdir(path)
        for temp in temps:
            samples[os.path.join(path, temp)] = one_person_samples
    print samples
    data = np.zeros((len(samples), dimension, IMAGE_SIZE, IMAGE_SIZE))
    for index, sample in enumerate(samples):
        print sample, samples[sample]
        # data[index, :, :, :] = preprocess.readManyDicom_sorted(sample, IMAGE_SIZE, dimension) * 0.00390625
        data[index, :, :, :] = preparation.process_data(preparation.load_scan(sample), IMAGESIZE=64, HM_SLICES=64) * 0.00390625
    print data.shape
    # print data.shape
    # # only for test LeNet
    # # data = data * 0.00390625
    net.blobs['data'].data[...] = data
    output = net.forward()
    features = output[LAST_LAYER_NAME]
    with open(save_features_file, "w") as f:
        for index, sample in enumerate(samples):
            des = sample
            for i in features[index].tolist():
                # des = "%s %s" % (sample, " ".join(features[index].tolist()))
                des = des + ' ' + str(i)
            print des
            f.write(des+"\n")
    # return features, samples, patch

def write_model_dir_features(caffemodel_source_dir, target_save_path, test_data_source, deploy_file, dimension, LAST_LAYER_NAME="feat"):
    for file_name in os.listdir(caffemodel_source_dir):
        path = os.path.join(caffemodel_source_dir, file_name)
        ordinary_predict_dataset(test_data_source, caffemodel=path, deploy_file=deploy_file, dimension=dimension, LAST_LAYER_NAME=LAST_LAYER_NAME, save_features_file=os.path.join(target_save_path, file_name+"_features.txt"), IMAGE_SIZE=64)

def read__file_and_output_accuracy(sequence_source, features_file):
    with open(features_file) as f:
        temp = f.readlines()
    samples = {}
    for t in temp:
        a = t.split(" ")
        one_feature = [float(x) for x in a[1:]]
        samples[a[0]] = one_feature
    with open(sequence_source) as f:
        temp = f.readlines()
    result = []
    key_ = ""
    for t in temp:
        a = t.split(" ")
        real_result = int(a[2])
        p1 = os.path.join("/home/hzzone/test", a[0])
        p2 = os.path.join("/home/hzzone/test", a[1])
        result.append((distance.cosine_distnace(samples[p1], samples[p2]), real_result))
        t1 = os.listdir(p1)[0]
        t2 = os.listdir(p2)[0]
        id1 = dicom.read_file(os.path.join(p1, t1)).PatientID
        id2 = dicom.read_file(os.path.join(p2, t2)).PatientID
        key_ += "%s %s %s %s %s\n" % (p1, p2, id1, id2, distance.cosine_distnace(samples[p1], samples[p2]))

    with open("temp1.txt", "w") as f:
        f.write(key_)
    x_values = pylab.arange(-1.0, 1.01, 0.001)
    y_values = []
    for threshold in x_values:
        correct = 0
        for r in result:
            if r[0] < threshold and r[1] == 0:
                correct = correct + 1
            elif r[0] >= threshold and r[1] == 1:
                correct += 1
        y_values.append(float(correct) / len(result))
    max_index = np.argmax(y_values)
    print features_file
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_values, y_values)
    plt.plot(x_values[max_index], y_values[max_index], '*', color='red',
             label="(%s, %s)" % (x_values[max_index], y_values[max_index]))
    plt.legend()
    # fig = plt.figure(0)
    # fig.savefig(jpg_file_name)
    # plt.close(0)
    plt.show()

def x1(source):
    with open(source) as f:
        temp = f.readlines()
    key = ""
    for t in temp:
        a = t.split(" ")
        real_result = int(a[2])
        p1 = os.path.join("/home/hzzone/test", a[0])
        p2 = os.path.join("/home/hzzone/test", a[1])
        t1 = os.listdir(p1)[0]
        t2 = os.listdir(p2)[0]
        id1 = dicom.read_file(os.path.join(p1, t1)).PatientID
        id2 = dicom.read_file(os.path.join(p2, t2)).PatientID
        key += "%s %s %s %s %s\n" % (a[0], a[1], id1, id2, real_result)
    with open("temp1.txt", "w") as f:
        f.write(key)

if __name__ == "__main__":
    # write_model_dir_features("/home/hzzone/determination-of-identity/ct-test/siamese/model", "/home/hzzone/determination-of-identity/ct-test/siamese/siamese_features", test_data_source="/home/hzzone/classifited/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/lenet_siamese.prototxt", dimension=64, LAST_LAYER_NAME="feat")
    for file_name in os.listdir("../ct-test/siamese/siamese_features"):
        read__file_and_output_accuracy("./temp.txt", os.path.join("./classfication_features", file_name), os.path.join("./im", file_name+".jpg"))
    # x1("./temp.txt")
    # read__file_and_output_accuracy("./temp.txt", "/home/hzzone/determination-of-identity/python/features/lenet_siamese_train_iter_1000.caffemodel_features.txt", "1.jpg")


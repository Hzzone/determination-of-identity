# -*- coding: utf-8 -*-
import lmdb
import numpy as np
import os
# caffe_root = '/home/hzzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
caffe_root = '/home/hzzone/determination-of-identity/C3D/C3D-v1.1'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import pylab
import matplotlib.pyplot as plt
import dicom
import test
import distance

# source: Test dataset
def ordinary_predict_dataset(source, caffemodel, deploy_file, LAST_LAYER_NAME, save_features_file, gpu_mode=True):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    for each_sample in os.listdir(source):
        p1 = os.path.join(source, each_sample)
        print p1
        im = np.load(p1)
        patient_id = im[1]
        study_date = im[2]
        im = im[0]
        # im = im[np.newaxis, :]
        net.blobs['data'].data[...] = im
        output = net.forward()
        print output
        features = output[LAST_LAYER_NAME]
        print features
        with open(save_features_file, "a") as f:
            f.write("%s %s %s\n" % (patient_id, study_date, " ".join(map(str, features[0].tolist()))))

def write_model_dir_features(caffemodel_source_dir, target_save_path, test_data_source, deploy_file, LAST_LAYER_NAME):
    for file_name in os.listdir(caffemodel_source_dir):
        path = os.path.join(caffemodel_source_dir, file_name)
        ordinary_predict_dataset(test_data_source, caffemodel=path, deploy_file=deploy_file, LAST_LAYER_NAME=LAST_LAYER_NAME, save_features_file=os.path.join(target_save_path, file_name+"_features.txt"))

def read_file_and_output_accuracy(features_file):
    with open(features_file) as f:
        temp = f.readlines()
    samples = {}
    for t in temp:
        a = t.split(" ")
        one_feature = [float(x) for x in a[2:]]
        samples["/".join(a[:2])] = one_feature
    with open("../data/test_diff_sequence.txt") as f:
        _diff = f.readlines()
    with open("../data/test_same_sequence.txt") as f:
            _same = f.readlines()
    _same = [x.split(" ") for x in _same]
    _diff = [x.split(" ") for x in _diff][:len(_same)]
    x_values = pylab.arange(-1.0, 1.0, 0.0001)
    max_accuracy = 0
    max_accuracy_threshold = 0
    for threshold in x_values:
        correct = 0
        for r in _same:
            d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
            if d >= threshold:
                correct = correct + 1
        for r in _diff:
            d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
            if d < threshold:
                correct = correct + 1
        acc = float(correct)/(len(_same)+len(_diff))
        if acc >= max_accuracy:
            max_accuracy = acc
            max_accuracy_threshold = threshold
    print features_file, max_accuracy_threshold, max_accuracy
    
    # y_values = []
    # for threshold in x_values:
    #     correct = 0
    #     for r in result:
    #         if r[0] < threshold and r[1] == 0:
    #             correct = correct + 1
    #         elif r[0] >= threshold and r[1] == 1:
    #             correct += 1
    #     y_values.append(float(correct) / len(result))
    # max_index = np.argmax(y_values)
    # print features_file
    # plt.title("threshold-accuracy curve")
    # plt.xlabel("threshold")
    # plt.ylabel("accuracy")
    # plt.plot(x_values, y_values)
    # plt.plot(x_values[max_index], y_values[max_index], '*', color='red',
    #          label="(%s, %s)" % (x_values[max_index], y_values[max_index]))
    # plt.legend()
    # plt.show()
    
def write_reslut(features, threshold, diff_sequence_file, _same_sequence_file):
	with open(features) as f:
		temp = f.readlines()
	samples = {}
	for t in temp:
		a = t.split(" ")
		one_feature = [float(x) for x in a[2:]]
		samples["/".join(a[:2])] = one_feature
	with open("../data/test_diff_sequence.txt") as f:
		_diff = f.readlines()
	with open("../data/test_same_sequence.txt") as f:
		_same = f.readlines()
	_same = [x.split(" ") for x in _same]
	_diff = [x.split(" ") for x in _diff]
	result = []
	for r in _same:
		d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
		if d >= threshold:
			r.append(1)
		else:
			r.append(0)
		result.append(r)
	for r in _diff:
		d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
		if d >= threshold:
			r.append(1)
		else:
			r.append(0)
		result.append(r)
	with open("test.txt") as f:
		for s in result:
			f.write("%s\n" % " ".join(s))
		

if __name__ == "__main__":
    # write_model_dir_features("/home/hzzone/determination-of-identity/ct-test/siamese/model", "/home/hzzone/determination-of-identity/ct-test/siamese/siamese_features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/3d_siamese_deploy.prototxt", LAST_LAYER_NAME="fc8")
    for file_name in os.listdir("../ct-test/siamese/siamese_features"):
	    read_file_and_output_accuracy(os.path.join("../ct-test/siamese/siamese_features", file_name))
    # x1("./temp.txt")
    # read__file_and_output_accuracy("./temp.txt", "/home/hzzone/determination-of-identity/python/features/lenet_siamese_train_iter_1000.caffemodel_features.txt", "1.jpg")


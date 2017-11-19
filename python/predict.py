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
import distance

# source: Test dataset
def ordinary_predict_dataset(source, caffemodel, deploy_file, LAST_LAYER_NAME, save_features_file, gpu_mode=True, is_3d=True):
    if gpu_mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    content = ""
    for each_sample in os.listdir(source):
        p1 = os.path.join(source, each_sample)
        print(p1)
        im = np.load(p1)
        patient_id = im[1]
        study_date = im[2]
        im = im[0]
        # im = im[np.newaxis, :]
        if is_3d:
	        im = im[np.newaxis, :]
        net.blobs['data'].data[...] = im
        output = net.forward()
        features = output[LAST_LAYER_NAME]
        content += "%s %s %s\n" % (patient_id, study_date, " ".join(map(str, features[0].tolist())))
	with open(save_features_file, "w") as f:
		f.write(content)

def write_model_dir_features(caffemodel_source_dir, target_save_path, test_data_source, deploy_file, LAST_LAYER_NAME, is_3d=True):
    for file_name in os.listdir(caffemodel_source_dir):
        path = os.path.join(caffemodel_source_dir, file_name)
        ordinary_predict_dataset(test_data_source, caffemodel=path, deploy_file=deploy_file, LAST_LAYER_NAME=LAST_LAYER_NAME, save_features_file=os.path.join(target_save_path, file_name+"_features.txt"), is_3d=is_3d)

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
    x_values = pylab.arange(0.99999999, 1.0, 0.00000000001)
    max_accuracy = 0
    max_accuracy_threshold = 0
    same_distance = []
    diff_distance = []
    for r in _same:
        same_distance.append(distance.cosine_distnace(samples[r[0]], samples[r[1]]))
    for r in _diff:
        diff_distance.append(distance.cosine_distnace(samples[r[0]], samples[r[1]]))
	# print same_distance
	# print diff_distance
    total = len(_same)+len(_diff)
    same_distance = np.array(same_distance)
    diff_distance = np.array(diff_distance)
    for threshold in x_values:
        s = np.sum(same_distance>=threshold) + np.sum(diff_distance<threshold)
        acc = float(s)/total
        if acc >= max_accuracy:
            max_accuracy = acc
            max_accuracy_threshold = threshold
    print(features_file, max_accuracy_threshold, max_accuracy)
    
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

def write(features):
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
	_same = [x.strip("\n").split(" ") for x in _same]
	_diff = [x.strip("\n").split(" ") for x in _diff]
	result = []
	for r in _same:
		d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
		r.append(str(d))
		result.append(r)
	for r in _diff:
		d = distance.cosine_distnace(samples[r[0]], samples[r[1]])
		r.append(str(d))
		result.append(r)
	with open("test.txt", "w") as f:
		for s in result:
			# print(s)
			# print(s[0], s[1], s[3])
			f.write("%s %s %s %s\n" % (s[0], s[1], s[2], s[3]))

if __name__ == "__main__":
	# write("../ct-test/siamese/siamese_features/c3d_lenet_iter_2400.caffemodel_features.txt")
	# write_model_dir_features("/home/hzzone/1tb/id-model/classification/3d/3d_alexnet", "/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_alexnet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_alexnet/deploy.prototxt", LAST_LAYER_NAME="fc7")
	# write_model_dir_features("/home/hzzone/1tb/id-model/classification/3d/3d_caffenet", "/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_caffenet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_caffenet/deploy.prototxt", LAST_LAYER_NAME="fc7")
	# write_model_dir_features("/home/hzzone/1tb/id-model/siamese/3d/3d_alexnet", "/home/hzzone/determination-of-identity/ct-test/siamese/3d/3d_alexnet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/3d/3d_alexnet/deploy.prototxt", LAST_LAYER_NAME="fc8")
	# write_model_dir_features("/home/hzzone/1tb/id-model/siamese/3d/3d_caffenet", "/home/hzzone/determination-of-identity/ct-test/siamese/3d/3d_caffenet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/3d/3d_caffenet/deploy.prototxt", LAST_LAYER_NAME="fc8")
	# write_model_dir_features("/home/hzzone/1tb/id-model/classification/2d/2d_alexnet", "/home/hzzone/determination-of-identity/ct-test/classification/2d/2d_alexnet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/classification/2d/2d_alexnet/deploy.prototxt", LAST_LAYER_NAME="fc7", is_3d=False)
	# write_model_dir_features("/home/hzzone/1tb/id-model/classification/2d/2d_caffenet", "/home/hzzone/determination-of-identity/ct-test/classification/2d/2d_caffenet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/classification/2d/2d_caffenet/deploy.prototxt", LAST_LAYER_NAME="fc7", is_3d=False)
	# write_model_dir_features("/home/hzzone/1tb/id-model/siamese/2d/2d_alexnet", "/home/hzzone/determination-of-identity/ct-test/siamese/2d/2d_alexnet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/2d/2d_alexnet/deploy.prototxt", LAST_LAYER_NAME="fc8", is_3d=False)
	# write_model_dir_features("/home/hzzone/1tb/id-model/siamese/2d/2d_caffenet", "/home/hzzone/determination-of-identity/ct-test/siamese/2d/2d_caffenet/features", test_data_source="/home/hzzone/1tb/id-data/test", deploy_file="/home/hzzone/determination-of-identity/ct-test/siamese/2d/2d_caffenet/deploy.prototxt", LAST_LAYER_NAME="fc8", is_3d=False)
	for file_name in os.listdir("/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_alexnet/features"):
		read_file_and_output_accuracy(os.path.join("/home/hzzone/determination-of-identity/ct-test/classification/3d/3d_alexnet/features", file_name))


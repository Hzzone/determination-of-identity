# import
import dicom
import matplotlib.pyplot as plt
import os
from os.path import join as ospj
import numpy as np
import cv2
from scipy.misc import bytescale
import math
caffe_root = '/home/hzzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
from itertools import combinations
import random
import lmdb
import json

train_source = "/home/hzzone/classifited/train"
test_source = "/home/hzzone/classifited/test"

def resize_from_path(source, IMAGE_SIZE=227):
    ds = dicom.read_file(source)
    pixel_array = ds.pixel_array
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    return im*0.00390625

def resize_from_ds(ds, IMAGE_SIZE=227):
    pixel_array = ds.pixel_array
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    return im*0.00390625

def resize_from_array(pixel_array, IMAGE_SIZE=227):
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    return im*0.00390625


# Load the scans in given folder path
# threshold: HU threshold
def load_scan(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # Convert to Hounsfield units (HU)
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    # all_scans = []
    # for each_slice in slices:
    #     intercept = each_slice.RescaleIntercept
    #     slope = each_slice.RescaleSlope
    #     image = each_slice.pixel_array
    #     if slope != 1:
    #         image[(slope * image + intercept)<threshold] = 0
    #     all_scans.append(image)
    # return np.array(all_scans)

    return np.array([each_slice.pixel_array for each_slice in slices])

def plot_one_scan(scan):
    plt.imshow(scan)
    plt.show()


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
#     f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(50, 50))
#     for i in range(0, scan.shape[0]):
#         plots[int(i / 20), int((i % 20) / 5)].axis('off')
#         plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
#     plt.show()
#     f, plots = plt.subplots(int(scan.shape[0] / 4) + 1, 4, figsize=(50, 50))
    f, plots = plt.subplots(int(scan.shape[0] / 4), 4, figsize=(100, 100))
    f.tight_layout()
    for i in range(0, scan.shape[0]):
        plots[int(i / 4), int((i % 4))].axis('off')
        plots[int(i / 4), int((i % 4))].imshow(cv2.resize(scan[i], (64, 64)), cmap=plt.cm.bone)
    plt.show()


def chunks(l, n, HM_SLICES):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    '''Yield successive n-sized chunks from l.'''
    count=0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count = count+1

def mean(l):
    return sum(l) / len(l)

def process_data(slices, IMAGESIZE=64, HM_SLICES=64):
    slices = [resize_from_array(each_slice, IMAGE_SIZE=IMAGESIZE) for each_slice in slices]
    # chunk_sizes = int(math.floor(len(slices) / HM_SLICES))
    new_slices = []
    # for slice_chunk in chunks(slices, chunk_sizes):
#         slice_chunk = list(map(mean, zip(*slice_chunk)))
#         new_slices.append(slice_chunk)
#     return np.array(new_slices)
    chunk_sizes = math.floor(len(slices) / HM_SLICES)

    for slice_chunk in chunks(slices, int(chunk_sizes), HM_SLICES):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    print(len(slices), len(new_slices))
    return np.array(new_slices)

def save_np_file(IMAGE_SIZE=64, HM_SLICES=64):
    train_data = []
    test_data = []
    for person in os.listdir(train_source):
        p1 = ospj(train_source, person)
        for each_sample in os.listdir(p1):
            p2 = ospj(p1, each_sample)
            print(p2)
            train_data.append([process_data(load_scan(p2)), person, each_sample])

    for person in os.listdir(test_source):
        p1 = ospj(test_source, person)
        for each_sample in os.listdir(ospj(test_source, person)):
            p2 = ospj(p1, each_sample)
            print(p2)
            test_data.append([process_data(load_scan(p2)), person, each_sample])

    np.save('../data/train_data-{}-{}-{}.npy'.format(IMAGE_SIZE, IMAGE_SIZE, HM_SLICES), train_data)
    np.save('../data/test_data-{}-{}-{}.npy'.format(IMAGE_SIZE, IMAGE_SIZE, HM_SLICES), train_data)


# generate dataset path
# combines the samples, return _same and _diff
def generate_siamese_dataset(source):
    all_samples = []
    _same = []
    _diff = []
    for person in os.listdir(source):
        person_dir = os.path.join(source, person)
        person_samples = os.listdir(person_dir)
        for each_person_sample in person_samples:
            # person: id, each_person_sample: study date
            all_samples.append([person, each_person_sample])
    sample_combinations = list(combinations(all_samples, 2))
    for one_combination in sample_combinations:
        # print one_combination
        one_combination = list(one_combination)
        if one_combination[0][0] == one_combination[1][0] and one_combination[0][1] != one_combination[1][1]:
            one_combination.append(1)
            _same.append(one_combination)
        else:
            one_combination.append(0)
            _diff.append(one_combination)
    random.shuffle(_diff)
    random.shuffle(_same)
    return _same, _diff

def write_sequence_file():
    _same, _diff = generate_siamese_dataset(train_source)
    with open("../data/train_same_sequence.txt", "w") as f:
        for s in _same:
            f.write("%s %s %s\n" % ("/".join(s[0]), "/".join(s[1]), s[2]))
    with open("../data/train_diff_sequence.txt", "w") as f:
        for s in _diff:
            f.write("%s %s %s\n" % ("/".join(s[0]), "/".join(s[1]), s[2]))
    _same, _diff = generate_siamese_dataset(test_source)
    with open("../data/test_same_sequence.txt", "w") as f:
        for s in _same:
            f.write("%s %s %s\n" % ("/".join(s[0]), "/".join(s[1]), s[2]))
    with open("../data/test_diff_sequence.txt", "w") as f:
        for s in _diff:
            f.write("%s %s %s\n" % ("/".join(s[0]), "/".join(s[1]), s[2]))


def load(same_sequence_file, diff_sequence_file):
    with open(same_sequence_file) as f:
        # print f.readlines()[0]
        _same = json.loads(str(f.readlines()[0]))
    with open(diff_sequence_file) as f:
        _diff = json.loads(f.readlines()[0])
    print _same
    print _diff

# source: the dataset folder
'''
pay attention to shuffle the train list
'''
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


if __name__ == "__main__":
    # plot_ct_scan(load_scan("/home/hzzone/classifited/train/0000279404/20150528"))
    # plot_ct_scan(process_data(load_scan("/home/hzzone/classifited/train/0000279404/20150528")))
    # slices = process_data(load_scan("/home/hzzone/classifited/train/0000279404/20150528"))
    # save_np_file()
    write_sequence_file()
    # load("../data/train_same_sequence.txt", "../data/train_diff_sequence.txt")

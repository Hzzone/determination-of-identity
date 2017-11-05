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
# import caffe
from itertools import combinations
import random
import lmdb
import utils
import h5py as hy


import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')

train_source = "/home/hzzone/classifited/train"
test_source = "/home/hzzone/classifited/test"

def resize_from_path(source, IMAGE_SIZE=227):
    ds = dicom.read_file(source)
    pixel_array = ds.pixel_array
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    # return im*0.00390625
    return im

def resize_from_ds(ds, IMAGE_SIZE=227):
    pixel_array = ds.pixel_array
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    # return im*0.00390625
    return im

def resize_from_array(pixel_array, IMAGE_SIZE=227):
    im = cv2.resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    # return im*0.00390625
    return im


# Load the scans in given folder path
# threshold: HU threshold
def load_scan(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # Convert to Hounsfield units (HU)
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    return np.array([utils.get_bone_pixel_array(each_slice) for each_slice in slices])

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
        # print map(mean, zip(*slice_chunk))
        new_slices.append(slice_chunk)

    logging.debug("%s %s" % (len(slices), len(new_slices)))
    return np.array(new_slices)

def save_np_file(IMAGE_SIZE=64, HM_SLICES=64):
    train_data = []
    test_data = []
    for person in os.listdir(train_source):
        p1 = ospj(train_source, person)
        for each_sample in os.listdir(p1):
            p2 = ospj(p1, each_sample)
            logging.debug(p2)
            train_data.append([process_data(load_scan(p2)), person, each_sample])

    for person in os.listdir(test_source):
        p1 = ospj(test_source, person)
        for each_sample in os.listdir(ospj(test_source, person)):
            p2 = ospj(p1, each_sample)
            logging.debug(p2)
            test_data.append([process_data(load_scan(p2)), person, each_sample])

    np.save('../data/train_data-{}-{}-{}.npy'.format(IMAGE_SIZE, IMAGE_SIZE, HM_SLICES), train_data)
    np.save('../data/test_data-{}-{}-{}.npy'.format(IMAGE_SIZE, IMAGE_SIZE, HM_SLICES), test_data)


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
        lines = f.readlines()
        same = [line.strip('\n').split(" ") for line in lines]
    with open(diff_sequence_file) as f:
        lines = f.readlines()
        diff = [line.strip('\n').split(" ") for line in lines]
    all_samples = []
    all_samples.extend(same)
    all_samples.extend(diff[:len(same)])
    random.shuffle(all_samples)
    random.shuffle(all_samples)
    return all_samples

# source: the dataset folder
'''
pay attention to shuffle the train list
'''
def generate_siamese_lmdb(same_sequence_file, diff_sequence_file, data_np_source, save_target, dimension, IMAGE_SIZE):
    env = lmdb.Environment(save_target, map_size=int(1e12))
    sequence_file = load(same_sequence_file, diff_sequence_file)
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 2*dimension
        datum.height = IMAGE_SIZE
        datum.width = IMAGE_SIZE
        sample = np.zeros((2*dimension, IMAGE_SIZE, IMAGE_SIZE))
        for index, sequence_sample in enumerate(sequence_file):
            label = int(sequence_sample[2])
            first_sample = sequence_sample[0]
            second_sample = sequence_sample[1]
            first_sample_id = first_sample.split("/")[0]
            first_sample_study_date = first_sample.split("/")[1]
            second_sample_id = second_sample.split("/")[0]
            second_sample_study_date = second_sample.split('/')[1]
            p1 = os.path.join(data_np_source, "%s_%s.npy" % (first_sample_id, first_sample_study_date))
            p2 = os.path.join(data_np_source, "%s_%s.npy" % (second_sample_id, second_sample_study_date))
            sample[:dimension] = np.load(p1)[0]
            sample[dimension:] = np.load(p2)[0]
            datum.data = sample.tobytes()
            datum.label = label
            str_id = "%8d" % index
            txn.put(str_id, datum.SerializeToString())
            logging.debug("%s %s" % (sequence_sample, label))

def generate_ordinary_hdf5(data_np_source, save_target, dimension=64, IMAGE_SIZE=64):
    samples = []
    for np_file in os.listdir(data_np_source):
        samples.append(np_file.split("_")[0])
    samples = set(samples)
    k = {}
    for index, s in enumerate(samples):
        k[s] = index
    print(k)
    print(len(k))
    # for index, np_file in enumerate(os.listdir(data_np_source)):
    #     p_id = np_file.split("_")[0]
    #     with hy.File(os.path.join(save_target, str(index)+".h5"), 'w') as h5_file:
    #         data = np.zeros((1, dimension, IMAGE_SIZE, IMAGE_SIZE))
    #         label = k[p_id]
    #         p = os.path.join(data_np_source, np_file)
    #         data[0] = np.load(p)[0]
    #         data.astype(np.float32)
    #         labels = np.zeros(1, dtype=np.float32)
    #         labels[0] = label
    #         h5_file['data'] = data
    #         h5_file['label'] = labels
    #         logging.debug(p)




def generate_siamese_hdf5(same_sequence_file, diff_sequence_file, data_np_source, save_target, dimension, IMAGE_SIZE, is_3d=True):
    sequence_file = load(same_sequence_file, diff_sequence_file)
    for index, sequence_sample in enumerate(sequence_file):
        with hy.File(os.path.join(save_target, str(index)+".h5"), 'w') as h5_file:
            data = np.zeros((1, 2*dimension, IMAGE_SIZE, IMAGE_SIZE))
            label = int(sequence_sample[2])
            first_sample = sequence_sample[0]
            second_sample = sequence_sample[1]
            first_sample_id = first_sample.split("/")[0]
            first_sample_study_date = first_sample.split("/")[1]
            second_sample_id = second_sample.split("/")[0]
            second_sample_study_date = second_sample.split('/')[1]
            p1 = os.path.join(data_np_source, "%s_%s.npy" % (first_sample_id, first_sample_study_date))
            p2 = os.path.join(data_np_source, "%s_%s.npy" % (second_sample_id, second_sample_study_date))
            data[0][:40] = np.load(p1)[0]
            data[0][40:] = np.load(p2)[0]
            if is_3d:
                data = data[np.newaxis, :]
            data.astype(np.float32)
            labels = np.zeros(1, dtype=np.float32)
            labels[0] = label
            h5_file['pair_data'] = data
            h5_file['sim'] = labels
            logging.debug(sequence_sample)

def generate_hdf5_txt(hdf5_source, save_path):
    with open(save_path, "w") as f:
        for root, dirs, files in os.walk(hdf5_source):
            for hdf5_file in files:
                f.write(os.path.join(root, hdf5_file)+"\n")


if __name__ == "__main__":
    '''
    data process test
    '''
    # plot_ct_scan(load_scan("/home/hzzone/classifited/train/0000279404/20150528"))
    # utils.plot_3d(load_scan("/home/hzzone/classifited/train/0000279404/20150528"))
    # utils.plot_3d(process_data(load_scan("/home/hzzone/classifited/train/0000279404/20150528")))
    # slices = process_data(load_scan("/home/hzzone/classifited/train/0000279404/20150528"))
    # print load_scan("/home/hzzone/classifited/train/0000279404/20150528")[0]
    # plot_one_scan(utils.get_bone_pixel_array(dicom.read_file("/home/hzzone/classifited/train/0000279404/20150528/41239121")))

    '''
    save file test
    '''
    # save_np_file()
    # array = np.load("../data/train_data-64-64-64.npy")
    # print array.shape
    # print array[0].shape
    # print array[0][0].shape
    # print array[0][1]
    # print array[0][2]
    # plot_ct_scan(array[0][0])
    # array = np.load("../data/test_data-64-64-64.npy")
    # print array.shape

    '''
    generate lmdb dataset
    '''
    # write_sequence_file()
    # print load("../data/train_same_sequence.txt", "../data/train_diff_sequence.txt")
    # generate_siamese_lmdb("../data/train_same_sequence.txt", "../data/train_diff_sequence.txt", "/home/hzzone/1tb/id-data/train", save_target="/home/hzzone/1tb/id-data/lmdb/siamese_train_lmdb", IMAGE_SIZE=300, dimension=200)
    # generate_siamese_lmdb("../data/test_same_sequence.txt", "../data/test_diff_sequence.txt", "/home/hzzone/1tb/id-data/test", save_target="/home/hzzone/1tb/id-data/lmdb/siamese_test_lmdb", IMAGE_SIZE=300, dimension=200)
    # generate_ordinary_lmdb("../data/test_data-64-64-64.npy", save_target="../data/ordinary_test_lmdb")
    # generate_ordinary_lmdb("../data/train_data-64-64-64.npy", save_target="../data/ordinary_train_lmdb")


    '''
    generate hdf5
    '''
    generate_siamese_hdf5("../data/train_same_sequence.txt", "../data/train_diff_sequence.txt", "/home/hzzone/1tb/id-data/train", save_target="/home/hzzone/1tb/id-data/hdf5/classification_train", IMAGE_SIZE=270, dimension=40, is_3d=False)
    generate_siamese_hdf5("../data/test_same_sequence.txt", "../data/test_diff_sequence.txt", "/home/hzzone/1tb/id-data/test", save_target="/home/hzzone/1tb/id-data/hdf5/classification_test", IMAGE_SIZE=270, dimension=40, is_3d=False)
    # generate_hdf5_txt("/home/hzzone/1tb/id-data/classification_hdf5/train", "train.txt")
    # generate_hdf5_txt("/home/hzzone/1tb/id-data/hdf5/test", "test.txt")
    # generate_ordinary_hdf5("/home/hzzone/1tb/id-data/train", save_target="/home/hzzone/1tb/id-data/classification_hdf5/train", IMAGE_SIZE=270, dimension=40)
    pass

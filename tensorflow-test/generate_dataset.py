import dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import math




def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(each_patient_sample_path, img_px_size=50, hm_slices=20, visualize=False):
    slices = [dicom.read_file(os.path.join(each_patient_sample_path, s)) for s in os.listdir(each_patient_sample_path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array), (img_px_size, img_px_size)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices + 2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if len(new_slices) == hm_slices + 1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num + 1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    return np.array(new_slices)


base_data_dir = '/Users/HZzone/Desktop/determination-data'
train_data_dir = os.path.join(base_data_dir, 'train')
test_data_dir = os.path.join(base_data_dir, 'test')
patients = os.listdir(train_data_dir)

IMG_SIZE_PX = 227
SLICE_COUNT = 20

#                                               stage 1 for real.
patients = os.listdir(train_data_dir)



much_data = []
for label, patient in enumerate(patients):
    try:
        path = os.path.join(train_data_dir, patient)
        for each_person_sample in os.listdir(path):
            temp = os.path.join(path, each_person_sample)
            img_data = process_data(temp, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
            # print(img_data.shape,label)
            much_data.append([img_data, label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT), much_data)



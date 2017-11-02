# -*- coding: utf-8 -*-
import dicom
# import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize
from scipy.misc import bytescale
import td_process

def show(source):
    ds = dicom.read_file(source)
    pixel_array = ds.pixel_array
    plt.imshow(pixel_array)
    plt.show()
    print(ds.PatientID)

def getPatientID(diretory, targetID="0008441186"):
    result = {}
    for root, dirs, files in os.walk(diretory):
        # 0008441186 115 17 1
        for dicom_file in files:
            path = os.path.join(root, dicom_file)
            ds = dicom.read_file(path)
            print(path + " " + ds.PatientID)
            result[ds.PatientID] = os.path.dirname(path)
            break
    for key in result:
        if key == targetID:
            return result[key]

def process(source, IMAGE_SIZE=227):
    ds = dicom.read_file(source)
    pixel_array = ds.pixel_array
    im = resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    return im

def process_sorted(ds, IMAGE_SIZE=227):
    pixel_array = ds.pixel_array
    im = resize(pixel_array, (IMAGE_SIZE, IMAGE_SIZE))
    im = bytescale(im)
    return im

# read one sample dicom folder and return the dimension*dimension matrix
def readManyDicom(source, IMAGE_SIZE=227, dimension=150):
    sample = np.zeros((dimension, IMAGE_SIZE, IMAGE_SIZE))
    for index, dicom_file in enumerate(os.listdir(source)):
        path = os.path.join(source, dicom_file)
        im = process(path, IMAGE_SIZE=IMAGE_SIZE)
        sample[index, :, :] = im
    return sample


# read one sample dicom folder and return the dimension*dimension matrix
def readManyDicom_sorted(source, IMAGE_SIZE=227, dimension=150):
    sample = np.zeros((dimension, IMAGE_SIZE, IMAGE_SIZE))
    slices = td_process.load_scan(source)
    # for index, dicom_file in enumerate(os.listdir(source)):
    #     path = os.path.join(source, dicom_file)
    #     im = process(path, IMAGE_SIZE=IMAGE_SIZE)
    #     sample[index, :, :] = im
    for index, ds in enumerate(slices):
        im = process_sorted(ds, IMAGE_SIZE=IMAGE_SIZE)
        sample[index, :, :] = im
    return sample


def shape(source):
    for root, dirs, files in os.walk(source):
        for dicom_file in files:
            print(dicom.read_file(os.path.join(root, dicom_file)).pixel_array.shape)


if __name__ == "__main__":
    shape("/Users/HZzone/Desktop/mdzz")
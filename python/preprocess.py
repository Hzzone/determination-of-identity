# -*- coding: utf-8 -*-
import dicom
import numpy
import matplotlib.pyplot as plt
import os

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



if __name__ == "__main__":
    # show("/Users/HZzone/Desktop/mdzz/03300000/00684232")
    # print getPatientID("/Volumes/Hzzone/data")
    # print("end "+getPatientID("/Volumes/Hzzone/same"))
    print("hello world")

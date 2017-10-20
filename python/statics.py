# -*- coding: utf-8 -*-
import dicom
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil


def move_same(source, target_dir="/Users/HZzone/Desktop/mdzz"):
    patient_ids = {}
    num = {}
    for root, dirs, files in os.walk(source):
        for dicom_file in files:
            path = os.path.join(root, dicom_file)
            ds = dicom.read_file(path)
            patient_id = ds.PatientID
            patient_ids[root] = patient_id
            if patient_id not in num.keys():
                num[patient_id] = 0
            num[patient_id] = num[patient_id] + 1
            break
    for key in num:
        if num[key] > 1:
            for path in patient_ids:
                if key == patient_ids[path]:
                    print(path)
                else:
                    continue
                target_path = os.path.join(target_dir, key)
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                target_path = os.path.join(target_path, os.path.split(path)[1])
                print(target_path+path)
                shutil.copytree(path, target_path)
            print("---------------")


def statics_slices(source):
    result = {}
    for root, dirs, files in os.walk(source):
        length = len(files)
        if length not in result.keys():
            result[length] = 0
        result[length] = result[length] + 1
    keys = tuple(result.keys())
    values = tuple(result.values())
    plt.figure(1)
    width = 1
    for i in range(len(result)):
        plt.figure(1)
        plt.bar(i * width, values[i], width)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    print(keys)
    print(values)


def removeExceedingSample(source, threshold=200):
    pass

def renameFolder(source):
    samples = os.listdir(source)
    for s in samples:
        file_name = np.random.randint(10000000, 100000000)
        os.rename(os.path.join(source, s), os.path.join(source, str(file_name)))
        print file_name

def delete_DS_Store(source):
    for root, dirs, files in os.walk(source):
        for file_name in files:
            if file_name == ".DS_Store":
                os.remove(os.path.join(root, file_name))


if __name__ == "__main__":
    # print(len(statics_same("/Volumes/Hzzone/data")))
    # statics_slices("/Users/HZzone/Desktop/mdzz")
    # statics_slices("/Volumes/Hzzone/same")
    # statics_same("/Volumes/Hzzone/data")
    # renameFolder("/Volumes/Hzzone/same")
    move_same("/Volumes/Hzzone/data", target_dir="/Users/HZzone/Desktop/temp")


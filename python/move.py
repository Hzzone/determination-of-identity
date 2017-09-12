import dicom
import os
import shutil
path = "/Volumes/Hzzone/same_person/DICOM/20170910"
save_path = "/Users/HZzone/Desktop/data"
for root, dirs, files in os.walk(path):
    for dicom_file in files:
        dicom_file = os.path.join(root, dicom_file)
        print dicom_file
        ds = dicom.read_file(dicom_file)
        save_one_person = os.path.join(save_path, ds.PatientID)
        if not os.path.exists(save_one_person):
            os.mkdir(save_one_person)
        save_one_person_one_sample = os.path.join(save_one_person, ds.StudyDate)
        if not os.path.exists(save_one_person_one_sample):
            os.mkdir(save_one_person_one_sample)
        shutil.copy(dicom_file, save_one_person_one_sample)
        print save_one_person_one_sample

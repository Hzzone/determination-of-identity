import glob
import os
import numpy as np # linear algebra
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
import cv2
from os.path import join as ospj
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')

'''
Preprocess steps:
1. convert pixel values to HU
2. resampling 1*1*1 pixel spacing
3. keep the head and discard the left things
4. normalize
5. zero centered
6. padding to 200*300*300
'''

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # print(image)
    

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        # 去除挡板
        # image[slice_number][mask] = 0

        # if slope != 1:
        # image[slice_number] = slope * image[slice_number].astype(np.float64)
        # image[slice_number] = image[slice_number].astype(np.int16)
            
        # image[slice_number] += np.int16(intercept)
        image[slice_number] = segment_head_mask(image[slice_number], slope, intercept)

    return np.array(image, dtype=np.int16)


# def resample(image, scan, slice_count, new_spacing=[1, 1, 1]):
#     # Determine current pixel spacing
#     spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
#
#     resize_factor = spacing / new_spacing
#     new_real_shape = image.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     new_shape[0] = slice_count
#     real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor
#
#     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
#
#     return image, new_spacing

'''
down-resampling to the size
to solve the problem:
    have the same number of slices scan and image size: eg: 40*120*120
    and decrease the pressure of computation
'''
def resample(image, expected_shape):

    real_resize_factor = np.array(expected_shape) / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    
    verts, faces, x, y = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
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

# 返回不等于background的最大面积的像素值
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_head_mask(image, slope, intercept):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(slope*image.astype(np.float64)+intercept > -400, dtype=np.int8)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    #   为了找到背景色，就用右上角第一个点
    #Fill the air around the person
#     labels[background_label == labels] = -1
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    # For every slice we determine the largest solid structure
    # for i, axial_slice in enumerate(binary_image):
    labels = measure.label(binary_image)
    background_label = labels[0, 0]
#             axial_slice = axial_slice - 1
#             binary_image -= 1
#             labeling = measure.label(binary_image)
    l_max = largest_label_volume(labels, bg=background_label)
    if l_max is not None: #This slice contains some lung
        labels[labels != l_max] = 0
#             labels[labels == -1] = 0
        labels[labels != 0] = 1
        # plt.imshow(image[i])
        # plt.show()
    return (image*labels)*slope + intercept

def padding(image, expected_shape=(300, 300)):
    dim, width, height = image.shape
    # print(image.shape)
    expected_width, expected_height = expected_shape
    padding_image = np.ones((dim, expected_height, expected_width))*(-1024)
    # print(padding_image.shape)
    # low_z_offset = int((expected_dim-dim)/2)
    # high_z_offset = int((expected_dim+dim)/2)
    # print(low_z_offset, high_z_offset)
    high_x_offset = int((expected_width+width)/2)
    low_x_offset = int((expected_width-width)/2)
    # print(low_x_offset, high_x_offset)

    low_y_offset = int((expected_height-height)/2)
    high_y_offset = int((expected_height+height)/2)
    # print(low_y_offset, high_y_offset)
    for index_dimmension in range(dim):
        padding_image[index_dimmension, low_y_offset: high_y_offset, low_x_offset: high_x_offset] = image[index_dimmension]
    # print(padding_image.shape)
    return padding_image
				
MIN_BOUND = -400.0
MAX_BOUND = 2000.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def calculate_pixel_mean():
	s = 0.
	source = ["/home/hzzone/classifited/train", "/home/hzzone/classifited/test"]
	file_list = []
	for file_source in source:
		for person in os.listdir(file_source):
			p1 = ospj(file_source, person)
			for each_sample in os.listdir(p1):
				p2 = ospj(p1, each_sample)
				file_list.append(p2)
	for each_sample in file_list:
		s += np.sum(preprocess(each_sample))
		logging.debug(each_sample)
	print(s)
	print(s/(len(file_list)*40*120*120))

def preprocess(source):
	patient = load_scan(source)
	patient_pixels = get_pixels_hu(patient)
	# pix_resampled, spacing = resample(patient_pixels, patient, 40, [1, 1, 1])
	pix_resampled = resample(patient_pixels, (40, 120, 120))
	# padding_image = padding(pix_resampled, expected_shape=(270, 270))
	# padding_image.astype(np.float32)
	normalized_image = normalize(pix_resampled)
	return normalized_image

# 测试
if __name__ == "__main__":
	train_source = "/home/hzzone/classifited/train"
	test_source = "/home/hzzone/classifited/test"

	file_list = []
	for person in os.listdir(test_source):
		p1 = ospj(test_source, person)
		for each_sample in os.listdir(p1):
			p2 = ospj(p1, each_sample)
			file_list.append(p2)
	for index, each_sample in enumerate(file_list):
		logging.debug("%s %s" % (index, p2))
		patient = each_sample.split("/")[-2]
		study_date = each_sample.split("/")[-1]
		np.save("/home/hzzone/1tb/id-data/test/%s_%s.npy" % (patient, study_date), [preprocess(p2), person, each_sample])

	file_list = []
	for person in os.listdir(train_source):
		p1 = ospj(train_source, person)
		for each_sample in os.listdir(p1):
			p2 = ospj(p1, each_sample)
			file_list.append(p2)
	for index, each_sample in enumerate(file_list):
		logging.debug("%s %s" % (index, p2))
		patient = each_sample.split("/")[-2]
		study_date = each_sample.split("/")[-1]
		np.save("/home/hzzone/1tb/id-data/train/%s_%s.npy" % (patient, study_date), [preprocess(p2), person, each_sample])

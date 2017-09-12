import dicom
import matplotlib.pyplot as plt
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def get_bone_pixel_aaray(path):
    ds = dicom.read_file(path)
    pixel_array = ds.pixel_array
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    height, width = pixel_array.shape
    for i in range(height):
        for j in range(width):
            pixel_value = pixel_array[i][j]
            if pixel_value == -2000:
                pixel_array[i][j] = 0
            if pixel_value * slope + intercept < 300:
                pixel_array[i][j] = 0
    return pixel_array

def plot_3d(image):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    verts, faces, x, y = measure.marching_cubes(p, 0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

path = "/Users/HZzone/Desktop/data/0000494780/20120320"
slices = [os.path.join(path, s) for s in os.listdir(path)]
bone_images = [get_bone_pixel_aaray(dicom_file) for dicom_file in slices]
plot_3d(bone_images)

import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import preparation

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

def get_bone_pixel_array(ds, hu_threshold=300):
    pixel_array = ds.pixel_array
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    pixel_array[pixel_array == -2000] = 0
    pixel_array[pixel_array * slope + intercept < hu_threshold] = 0

    return pixel_array

def statistics_pixel(pixel_array):
    plt.hist(pixel_array.flatten(), bins=80, color='c')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    ds = dicom.read_file("/home/hzzone/classifited/train/0000279404/20150528/41239121")
    pixel_array = ds.pixel_array
    # print pixel_array
    # print get_bone_pixel_array(ds)
    # statistics_pixel(pixel_array)
    # statistics_pixel(get_bone_pixel_array(ds))
    # statistics_pixel(preparation.resize_from_array(get_bone_pixel_array(ds), IMAGE_SIZE=64))
    preparation.resize_from_array(preparation.load_scan("/home/hzzone/classifited/train/0000279404/20150528")[0])
    statistics_pixel(preparation.process_data(preparation.load_scan("/home/hzzone/classifited/train/0000279404/20150528"))[0])

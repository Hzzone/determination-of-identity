import dicom
import matplotlib.pyplot as plt

ds = dicom.read_file("/Users/HZzone/Desktop/data/0000494780/20120320/12782473")
pixel_array = ds.pixel_array
plt.subplot(1, 2, 1)
plt.imshow(pixel_array)
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
plt.subplot(1, 2, 2)
plt.imshow(pixel_array)
plt.show()

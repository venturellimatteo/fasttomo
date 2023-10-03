import numpy as np
import PIL
from PIL import Image
import os
import pylab as plt
import scipy.ndimage
from scipy.ndimage import median_filter
import skimage
import math
from scipy.ndimage.filters import sobel
import fabio.tifimage as tif
import SimpleITK as sitk

def SegConnectedThreshold(vol, val_min, val_max, seedListToSegment):
    ITK_Vol = imageFromNumpyToITK(vol)


    segmentationFilter = sitk.ConnectedThresholdImageFilter()

    for seed in seedListToSegment:
        seedItk = (seed[0], seed[1], seed[2])
        segmentationFilter.AddSeed(seedItk)

    segmentationFilter.SetLower(val_min)
    segmentationFilter.SetUpper(val_max)
    segmentationFilter.SetReplaceValue(1)
    ITK_Vol = segmentationFilter.Execute(ITK_Vol)
    image = imageFromITKToNumpy(ITK_Vol)

    return image.astype(np.uint8)

def imageFromNumpyToITK(vol):
    return sitk.GetImageFromArray(vol)


def imageFromITKToNumpy(vol):
    return sitk.GetArrayFromImage(vol)

def image_segmentation(image, val_min, val_max):
    seedListToSegment = [[252, 428,0], [428, 252,0]]
    segmented_1 = SegConnectedThreshold(image, val_min, val_max, seedListToSegment)
    im_1 = np.multiply(segmented_1, image)
    #im_1 = np.multiply(im_1, -1)


    return im_1

def saveTiff16bit(data, filename, minIm=0, maxIm=0, header=None):
	if (minIm == maxIm):
		minIm = np.amin(data)
		maxIm = np.amax(data)
	datatoStore = 65535.0 * (data - minIm) / (maxIm - minIm)
	datatoStore[datatoStore > 65535.0] = 65535.0
	datatoStore[datatoStore < 0] = 0

	datatoStore = np.asarray(datatoStore, np.uint16)


	if (header != None):
		tif.TifImage(data=datatoStore, header=header).write(filename)
	else:
		tif.TifImage(data=datatoStore).write(filename)

def center_rotation(img, dim_score, radius):
    off = (img.shape[0] - dim_score) // 2  # to center the score matrix in the middle of the image
    score = np.ones_like(img)

    for i in range(dim_score):
        for j in range(dim_score):
            region = img[off + i - radius:off + i + radius,
                     off + j - radius:off + j + radius]  # crop a region centered in i,j
            rot_region = np.rot90(region, 1)
            L2 = np.mean(np.sqrt((region - rot_region) ** 2))
            score[off + i, off + j] = L2
            center = np.unravel_index(np.argmin(score, axis=None), score.shape)

    return center


def shift_array_to_center(array, point_position):
    rows, cols = array.shape
    center_x = rows // 2
    center_y = cols // 2
    current_x, current_y = point_position

    displacement_x = center_x - current_x
    displacement_y = center_y - current_y

    shifted_array = np.zeros_like(array)

    for i in range(rows):
        for j in range(cols):
            shifted_i = i + displacement_x
            shifted_j = j + displacement_y
            if 0 <= shifted_i < rows and 0 <= shifted_j < cols:
                shifted_array[shifted_i, shifted_j] = array[i, j]

    return shifted_array

def find_center_of_mass(array):
    # Calculate the mass distribution along the x and y axes
    mass_x = np.sum(array ** 2, axis=0)
    mass_y = np.sum(array ** 2, axis=1)

    # Calculate the center of mass along the x and y axes
    center_x = np.sum(np.arange(array.shape[1]) * mass_x) / np.sum(mass_x)
    center_y = np.sum(np.arange(array.shape[0]) * mass_y) / np.sum(mass_y)
    print(center_x, center_y)
    return center_x, center_y

def create_mask(img, max_r):
    shape = img.shape
    mask = np.zeros_like(img)

    # Define the center coordinates of the circle
    center_x, center_y = (img.shape[0] // 2, img.shape[0] // 2)

    # Generate the circle by setting the appropriate array elements to 1
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance <= max_r:
                mask[i, j] = True
            else:
                mask[i, j] = False

    return mask

def find_angle(p1, p2):
    y1 = p1[1]
    y2 = p2[1]

    x1 = p1[0]
    x2 = p2[0]

    alpha = np.arctan((y2 - y1) / (x2 - x1))
    alpha = 180 * alpha / np.pi

    if (x2 - x1) < 0:
        alpha = alpha + 180

    return alpha


def rotate_array_around_point(array, center, angle):
    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Get the dimensions of the array
    rows, cols = array.shape

    # Create a new array to store the rotated elements
    rotated_array = np.zeros_like(array)

    # Iterate through each element in the array
    for i in range(rows):
        for j in range(cols):
            # Calculate relative coordinates of the element with respect to the center of rotation
            rel_x = j - center[0]
            rel_y = i - center[1]

            # Apply rotation transformation
            rotated_x = rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad)
            rotated_y = rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad)

            # Convert back to absolute coordinates by adding the center of rotation
            abs_x = int(rotated_x + center[0])
            abs_y = int(rotated_y + center[1])

            # Update the rotated_array with the new rotated positions
            if 0 <= abs_x < cols and 0 <= abs_y < rows:
                rotated_array[i, j] = array[abs_y, abs_x]

    return rotated_array

def radial_gradient(array):
    # Calculate the image gradients in the x and y directions
    gradient_x = sobel(array, axis=1, mode='constant')
    gradient_y = sobel(array, axis=0, mode='constant')

    # Compute the radial gradient as the magnitude of the gradients
    radial_gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return radial_gradient


def find_rotation(image):
    # Find center of image
    image = Image.open(image)
    img = np.array(image)
    img = np.interp(img, (img.min(), img.max()), (1, 0))
    center = (353,348)
    img2 = shift_array_to_center(img, center)
    #plt.imshow(img2)
    #plt.show()
    segmented_image = image_segmentation(img2,0.3,20)
    #plt.imshow(segmented_image)
    #plt.show()

    mask = create_mask(segmented_image, 70)
    img2 = segmented_image * mask

    img2 = median_filter(img2, size=5)
    mask = create_mask(img2, 65)
    img2 = img2 * mask

    img2[img2<=0.32] = 0
    img2[img2 > 0.32] = 1


    labels = scipy.ndimage.label(img2)

    max_size_label = -1
    label_max = -1

    if labels[1] != 1:
        for label in range(1,labels[1]):
            size = np.sum(labels[0] == label)
            if size>max_size_label:
                max_size_label =size
                label_max = label

        img2 = labels[0]
        img2[img2 != label_max] = 0
    #plt.imshow(img2)
    #plt.show()

    cm = find_center_of_mass(img2)
    alpha = find_angle(center, cm)

    return alpha, center

def rotate_one_image(im, alpha,center):
    im = Image.open(im)
    f = np.array(im)
    f = shift_array_to_center(f, center)
    rot_im = scipy.ndimage.rotate(f, alpha, axes=(1, 0))
    return rot_im

def rotate_volume(im_list, alpha, center, output_path):
    print(output_path)
    vol = np.zeros((len(im_list), 640, 640))
    # Rotate all and create a new dataset
    for i, im in enumerate(im_list):
        print('Image ' + str(i))
        im = Image.open(im)
        f = np.array(im)
        sa = shift_array_to_center(f, center)
        vol[i, :, :] = sa
    rot_vol = scipy.ndimage.rotate(vol, alpha, axes =(1,0))
    file_name = output_path + 'volume.npy'
    print(file_name)
    print('Saving volume ' + str(i))
    np.save(file_name, rot_vol)

    return rot_vol



if __name__ == '__main__':
    path = '/data/projects/whaitiri/Data/Data_Processing_July2022/Reconstructions/'
    out_path =  '/data/projects/whaitiri/Data/Data_Processing_July2022/Rotated_datasets/'

#'U:\\whaitiri\\Data\\Data_Processing_July2022\\Reconstructions\\'
#'U:\\whaitiri\\Data\\Data_Processing_July2022\\Rotated_datasets\\'

    for dataset in os.listdir(path):
        if ('P28A_FT_H_Exp3_3'in dataset):
            data_path = path + dataset + '\\'
            path_out = out_path + dataset +'\\'
            if not os.path.isdir(path_out):
                os.mkdir(path_out)
            for j, timestamp in enumerate(os.listdir(data_path)):
                time_path = data_path + timestamp + '/'
                path_out = path_out + timestamp + '/'
                if not os.path.isdir(path_out):
                    os.mkdir(path_out)
                list_images = []
                for i, image in enumerate(os.listdir(time_path)):
                    if ('.tiff' in image):
                        im_path = time_path + image
                        list_images.append(im_path)

                if j == 0:
                    print('Finding rotation angle')
                    path_conf = out_path + dataset + '/' + 'Check_Rotation'
                    if not os.path.isdir(path_conf):
                        os.mkdir(path_conf)
                    tiff_path = path_conf + '/' + 'image50.tiff'
                    im = Image.open(list_images[50])
                    f = np.array(im)
                    saveTiff16bit(f, path_conf)
                    rotation_angle, center = find_rotation(list_images[50])
                    rot_im = rotate_one_image(list_images[50], rotation_angle,center)
                    tiff_path = path_conf + '/' + 'rot_image50.tiff'
                    saveTiff16bit(rot_im, path_conf)

                print('Rotating all images')
                rotate_volume(list_images, rotation_angle, center, path_out)

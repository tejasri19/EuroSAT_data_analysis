import os
import numpy as np
import rasterio

# set the path to the directory containing the EuroSAT dataset
data_dir = '/home/tejasri/Desktop/eurosat-ms/Data/'

# set the path to the directory where the npy files will be saved
output_dir = '/home/tejasri/Desktop/eurosat-ms/'

# define a list of the class names in the dataset
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# loop over each class
for i in range(len(class_names)):
    # get the name of the current class
    class_name = class_names[i]

    # set the path to the directory containing the images for the current class
    class_dir = os.path.join(data_dir, class_name)

    # get a list of the filenames of the images for the current class
    image_filenames = os.listdir(class_dir)

    # initialize an empty array to hold the image data
    X = np.zeros((len(image_filenames), 64, 64, 13))

    # loop over each image in the current class
    for j in range(len(image_filenames)):
        # get the filename of the current image
        image_filename = image_filenames[j]

        # set the path to the current image
        image_path = os.path.join(class_dir, image_filename)
        
        #print(image_path)


        # open the image using rasterio
        with rasterio.open(image_path) as src:
            # read the image data as a numpy array
            image = src.read()

        # transpose the image data to the shape (13, 64, 64)
        image = np.transpose(image, (1, 2, 0))
        #np.float32(img)


        # add the image data to the X array
        X[j,:,:,:] = image

    # save the X array as an npy file
    np.save(os.path.join(output_dir, f'X{i}.npy'), X)

    # create an array of labels for the current class
    y = np.ones(len(image_filenames)) * i

    # save the y array as an npy file
    np.save(os.path.join(output_dir, f'y{i}.npy'), y)


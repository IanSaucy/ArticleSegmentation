"""
Author: Alex Thomas
Creation Date: 10/23/2020
Purpose: Exploratory Data Analysis on the BPL Liberator Articles
"""

# t-SNE for data visualization and dimensionality reduction
from tsne_python.tsne_python.tsne import tsne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# File Directory Methods (get_files_from_dir() method)
from os import listdir
from os.path import isfile, join
from pathlib import Path

# Image Loading/Saving Methods (load_image_from_file() method)
from PIL import Image

# Converting To Byte Stream Methods ()
import io

# Type Checking
from typing import Optional, List, Tuple

################## FUNCTIONS ##################

def get_files_from_dir(path: str, file_exts: Optional[List[str]]=['JPG']) -> Tuple[str,str]:
    """Goes through the directory and all subdirectories recursively and finds the files of the file_extensions you specified

    Args:
        path (str): The path to the directory you will be searching image files for
        file_exts (List[str]): A list of allowed extensions
            (default is ['JPG'])

    Returns:
        Tuple[str,str]: a tuple containing the path to the file, and the filename + extension
    """
    if file_exts is not None:
        file_exts = ['.' + file_ext.lower() for file_ext in file_exts]

    for f in listdir(path):
        if isfile(join(path, f)) and (not file_exts or Path(f).suffix in file_exts):
            yield (path,f)

def load_image_from_file(file_path: str) -> Image:
    """Loads the image file as a PIL.Image Object

    Args:
        file_path (str): The full path to find the image file

    Returns:
        Image: A PIL.Image object
    """
    return Image.open(file_path)    

def image_to_byte_array(image: Image) -> io.BytesIO:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def image_files_to_np_arrays(path: Optional[str] = None, image_paths: Optional[List[str]] = None, img_exts: Optional[List[str]] = ['JPG']) -> np.array:
    if not (path or image_paths):
        print('Need to provide a directory to get all images or a list of image_paths')
        return

    if (path and image_paths):
        print('Will only use one of these variables at a time! We will be using the path...')        

    # If user provided a directory to search image files from
    if path:

        image_paths = get_files_from_dir(path,file_exts=img_exts)
        for path, file_name in image_paths:
            img_path = path + file_name                    
            yield np.asarray(load_image_from_file(img_path))

    # If user provided a list of paths to images they want to parse
    elif image_paths:

        img_exts = ['.' + img_ext.lower() for img_ext in img_exts]
        for image_path in image_paths:

            if Path(image_path).suffix in img_exts:                
                yield np.asarray(load_image_from_file(image_path))

def image_files_to_byte_arrays(path: Optional[str] = None, image_paths: Optional[List[str]] = None, img_exts: Optional[List[str]] = ['JPG']) -> io.BytesIO:
    if not (path or image_paths):
        print('Need to provide a directory to get all images or a list of image_paths')
        return

    if (path and image_paths):
        print('Will only use one of these variables at a time! We will be using the path...')        

    # If user provided a directory to search image files from
    if path:

        image_paths = get_files_from_dir(path,file_exts=img_exts)
        for path, file_name in image_paths:
            img_path = path + file_name
            image = load_image_from_file(img_path)
            image_bytes = image_to_byte_array(image)
            yield image_bytes

    # If user provided a list of paths to images they want to parse
    elif image_paths:

        img_exts = ['.' + img_ext.lower() for img_ext in img_exts]
        for image_path in image_paths:

            if Path(image_path).suffix in img_exts:
                image = load_image_from_file(image_path)
                image_bytes = image_to_byte_array(image)
                yield image_bytes

def clean_image(image: Image, path: str, size: Optional[Tuple[int,int]]=(6027,8191), dpi: Optional[Tuple[int,int]]=(300, 300)) -> Image:
    """
    Rescaling image to 300dpi and resizing to the specified size (Default: (MaxWidth,MaxHeight))
    :param image: An image
    :return: A rescaled image with the set DPI
    """
    name = image.filename.rsplit('/',1)[-1]
    image = image.resize(size)
    image.save(join(path,name), dpi=dpi)
    return image

################## PLAYING WITH DATA ##################

# The path to the images I will be doing EDA on
path = './output_images/'

# Gather count of data
print('Finding Image Paths...')
image_paths = get_files_from_dir(path)
image_paths_list = list(image_paths)
print('Found All Image Paths!')
print('Image Count: ' + str(len(image_paths_list)))

# Download Image Data as PIL.Image
print('Loading Images...')
images = [load_image_from_file(path + filename) for path, filename in image_paths_list]
print('Loaded Images!')

# Store all the useful data of images in an array (width,height,format,mode,name)
img_data = [(img.size[0],img.size[1],img.format,img.mode,img.filename,img.info) for img in images] # L -> Greyscale

# Default Initialization of min and max size
maxWidth, maxHeight = images[0].size
minWidth = maxWidth
minHeight = maxHeight
summedWidth = 0
summedHeight = 0

maxWidthImage = minWidthImage = maxHeightImage = minHeightImage = None

# Calculating min and max for width and heights of images
for image in images:

    # Current Image's Dimensions
    width, height = image.size

    if width > maxWidth: maxWidthImage = image
    if width < minWidth: minWidthImage = image
    if height > maxHeight: maxHeightImage = image
    if height < minHeight: minHeightImage = image

    summedWidth += width
    summedHeight += height
    maxWidth = max(maxWidth, width)
    minWidth = min(minWidth, width)
    maxHeight = max(maxHeight, height)
    minHeight = min(minHeight, height)

avgWidth = summedWidth / len(images)
avgHeight = summedHeight / len(images)

txt = "Max Width: {maxWidth}, Min Width: {minWidth}, Max Height: {maxHeight}, Min Height: {minHeight}, Average Width: {avgWidth:.2f}, Average Height: {avgHeight:.2f}, Average Width to Height Ratio: {avgWidthHeightRatio:.2f}!"
print(txt.format(maxWidth=maxWidth,minWidth=minWidth,maxHeight=maxHeight,minHeight=minHeight,avgWidth=avgWidth,avgHeight=avgHeight,avgWidthHeightRatio=(avgWidth/avgHeight)))
print('Max Width Image: ' + str(maxWidthImage) + ', File: ' + str(maxWidthImage.filename))
print('Min Width Image: ' + str(minWidthImage) + ', File: ' + str(minWidthImage.filename))
print('Max Height Image: ' + str(maxHeightImage) + ', File: ' + str(maxHeightImage.filename))
print('Min Height Image: ' + str(minHeightImage) + ', File: ' + str(minHeightImage.filename))

# Seeing Info
print('Image 0\'s Info:' + str(img_data[0][-1]))

# Download clean images
output_path = './cleaned_images/'
for image in images:
    clean_image(image,output_path)

# Plot Image Data
plt.scatter([data[0] for data in img_data],[data[1] for data in img_data]) # Width, Height
plt.title('Height vs Width (Pixels)')
plt.xlabel('Width (Pixels)')
plt.ylabel('Height (Pixels)')
plt.show()

plt.scatter([x for x in range(len(img_data))],[data[-1]['jfif_unit'] for data in img_data]) # Width, Height
plt.title('Units for JFIF Density (Pixels)')
plt.xlabel('Image Index')
plt.ylabel('Unit (0: no units, 1: PPI, 2: PPC)')
plt.show()

# We must resize the images to ensure:
#   1. Same Dimensions to pass into a model/t-SNE
#   2. Efficiency because we don't want to work on gigantic images
# But since we are doing OCR later we need to ensure good DPI: https://abbyy.technology/en:kb:images_resolution_size_ocr
#   1. For regular texts (font size 8-10 points) it is recommended to use 300 dpi resolution for OCR
#   2. If scans have a smaller resolution, for example 200 dpi, then 10 point font will be too small. To compensate the “missing” pixels, the image will be scaled internally (up to 400 dpi).
#   3. For smaller font text sizes (8 points or smaller) we recommend to use A 400-600 dpi resolution.
# Current Technologies use/are capable of:
#   1. Currently ABBYY products can open images formats up to 32512*32512 pixels.
#   2. ABBYY Technologies use colour information for detecting areas and objects on the image.
#   3. So, if complex layouts have to be processed, it is recommend to use colour or at least, grey scale images

# Running t-SNE (Currently Not Working) (Requires images to have the same dimensions and for us to flatten them)
"""
numpy_array_generator = image_files_to_np_arrays(path=path)
curr = next(numpy_array_generator, None)
count = 0
limit = 2

X = np.array([], ndmin=2)

while curr is not None and count < limit:
    np.append(X, curr.flatten().reshape(1000, 12288), axis=1)
    curr = next(numpy_array_generator)
    count += 1

print(len(X))
print(X.shape)

Y = tsne(X, 2, 50, 20.0)
plt.scatter(Y[:, 0], Y[:, 1], 20)
plt.show()
"""
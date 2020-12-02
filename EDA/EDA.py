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

# To run kernel-PCA on our images to compress/reduce the dimensionality
from sklearn.decomposition import KernelPCA

# File Directory Methods (get_files_from_dir() method)
import os
# from os import listdir
# from os.path import isfile, join
from pathlib import Path

# For calling other scripts
import subprocess

# Image Loading/Saving Methods (load_image_from_file() method)
from PIL import Image

# Converting To Byte Stream Methods ()
import io

# Type Checking
from typing import Optional, List, Tuple

# Config
from config import config

############# Important Variables #############

listdir = os.listdir
isfile = os.path.isfile
join = os.path.join

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

"""
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
# img_data = [(img.size[0],img.size[1],img.format,img.mode,img.filename,img.info) for img in images] # L -> Greyscale

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
print('Image 0\'s Info:' + str(images[0].info))
"""

# Download clean images (Comment Out if you don't want to download)
"""
output_path = './cleaned_images/'
while len(images) > 0:
    image = images.pop(0)
    clean_image(image,output_path)
    image.close()
"""

"""
# Plot Image Data
# plt.scatter([data[0] for data in img_data],[data[1] for data in img_data]) # Width, Height
plt.scatter([image.size[0] for image in images],[image.size[1] for image in images]) # Width, Height
plt.title('Height vs Width (Pixels)')
plt.xlabel('Width (Pixels)')
plt.ylabel('Height (Pixels)')
plt.show()

# plt.scatter([x for x in range(len(img_data))],[data[-1]['jfif_unit'] for data in img_data]) # Units for pixels per space
plt.scatter([x for x in range(len(images))],[image.info['jfif_unit'] for image in images])
plt.title('Units for JFIF Density (Pixels)')
plt.xlabel('Image Index')
plt.ylabel('Unit (0: no units, 1: PPI, 2: PPC)')
plt.show()

# We must resize the images to ensure:
#   1. Same Dimensions to pass into a model/t-SNE
#   2. Efficiency because we don't want to work on gigantic images (But we actually do want gigantic images for OCR)
# But since we are doing OCR later we need to ensure good DPI: https://abbyy.technology/en:kb:images_resolution_size_ocr
#   1. For regular texts (font size 8-10 points) it is recommended to use 300 dpi resolution for OCR
#   2. If scans have a smaller resolution, for example 200 dpi, then 10 point font will be too small. To compensate the “missing” pixels, the image will be scaled internally (up to 400 dpi).
#   3. For smaller font text sizes (8 points or smaller) we recommend to use A 400-600 dpi resolution.
# Current Technologies use/are capable of:
#   1. Currently ABBYY products can open images formats up to 32512*32512 pixels.
#   2. ABBYY Technologies use colour information for detecting areas and objects on the image.
#   3. So, if complex layouts have to be processed, it is recommend to use colour or at least, grey scale images
"""

# Set OS Environment Variables
"""
WINDOWS:
set ABBYY_APPID=YourApplicationId
set ABBYY_PWD=YourPassword

UNIX:
export ABBYY_APPID=YourApplicationId
export ABBYY_PWD=YourPassword
"""
os.environ['ABBYY_APPID'] = config['ABBYY']['AppID']
os.environ['ABBYY_PWD'] = config['ABBYY']['AppPassword']
os.environ['ServerUrl'] = config['ABBYY']['ServerUrl']

"""
# Run ABBYY OCR on "unclean", "clean", and filtered data to get the results
pythonExecutable = 'python'                 # Consider changing 'python' to sys.executable
pythonProgPath = './ABBYY/process.py'
imageToProcess = './output_images/8k71pf49w_8k71pf50n.jpg'
outputFile = 'result1.txt'
args = '{pythonExecutable} {pythonProgPath} {imageToProcess} {outputFile}'.format(
    pythonExecutable=pythonExecutable,
    pythonProgPath=pythonProgPath,
    imageToProcess=imageToProcess,
    outputFile=outputFile).split() 
subprocess.call(args, shell=True) # Starts the process.py script which runs the AbbyyOnlineSDK, on a shell

imageToProcess = './cleaned_images/8k71pf49w_8k71pf50n.jpg'
outputFile = 'result2.txt'
args = '{pythonExecutable} {pythonProgPath} {imageToProcess} {outputFile}'.format(
    pythonExecutable=pythonExecutable,
    pythonProgPath=pythonProgPath,
    imageToProcess=imageToProcess,
    outputFile=outputFile).split() 
subprocess.call(args, shell=True) # Starts the process.py script which runs the AbbyyOnlineSDK, on a shell

# Get rid of all images
for i in range(len(images)):
    image = images.pop()
    image.close()

images = []
"""

"""
# Running t-SNE (Currently Not Working) (Requires images to have the same dimensions and for us to flatten them)

# The path to the cleaned images I will be running t-SNE on
path = './output_images/'

# Gather count of data
print('Finding Clean Image Paths...')
image_paths = get_files_from_dir(path)
# image_paths_list = list(image_paths)
print('Found All Clean Image Paths!')
# print('Image Count: ' + str(len(image_paths_list)))

# Download Image Data as PIL.Image
print('Loading Clean Images...')
images = []
LIMIT = 10
count = 0
curr = next(image_paths)

STANDARD_SIZE = (6027,8191) # Width, Height
NEW_WIDTH = int(2000)
NEW_HEIGHT = int((STANDARD_SIZE[1] / STANDARD_SIZE[0]) * NEW_WIDTH)
NEW_DIMENSIONS = (NEW_WIDTH,NEW_HEIGHT)
# NEW_DIMENSIONS = (256,256)

# SVD (Singular Value Decomposition) it does essentially the same thing as PCA finding eigen vectors and ranking them by most important
# https://medium.com/@rameshputalapattu/jupyter-python-image-compression-and-svd-an-interactive-exploration-703c953e44f6
def compress_svd(image,k):
    U,s,V = np.linalg.svd(image,full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return reconst_matrix,s

while curr is not None and count < LIMIT:
    path, filename = curr    
    # Resize to smaller images, and Black&White for t-SNE  
    image = load_image_from_file(path + filename)    
    image = image.convert('L')
    # image = image.convert('L').resize(NEW_DIMENSIONS)
    images.append(image)
    count += 1
print('Loaded Clean Images!')

# Show one of the images
# images[0].show()

# https://stackoverflow.com/questions/56491301/how-to-know-if-the-image-data-set-is-linearly-separable-or-not
# I would argue that most image datasets are linearly separable but the separation is useless. Because images usually live in a high-dimensional space, and as long as you have more features than samples everything is linearly separable (with probability 1)
# Gaussian radius basis function (RBF) kernel that is used to perform nonlinear dimensionality reduction via BF kernel principal component analysis (kPCA)
desired_size = (256, 256)
number_of_components = desired_size[0] * desired_size[1]
print('Number of components: ' + str(number_of_components))
transformer = KernelPCA(n_components=number_of_components, kernel='rbf') # Components being eigen vectors kept (Eigen vectors sorted in descending order of importance)

k = 150
nparrays = [] # Convert image to Numpy array, then flatten to a 1D list
for image in images:
    # compressed_image_array, s = compress_svd(np.asarray(image),k)
    # compressed_image = Image.fromarray(compressed_image_array)
    # print(compressed_image_array.shape)
    # compressed_image.show()
    nparrays.append(np.asarray(image).flatten())
    # nparrays.append(np.asarray(image))
    image.close()

X = np.array(nparrays)
images = []
print(X.shape)
print("X: %d bytes" % X.size * X.itemsize)
print()

# To apply PCA on image data, the images need to be converted to a one-dimensional vector representation using, for example, NumPy’s flatten() method.
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)
print("X_transformed: %d bytes" % (X_transformed.size * X_transformed.itemsize))
print()

# We need to reshape data if we want to convert it back into an image
X_transformed.reshape(desired_size)
print(X_transformed.shape)

# numpy_array_generator = image_files_to_np_arrays(path=path)
# curr = next(numpy_array_generator, None)
# count = 0
# limit = 2

# X = np.array([], ndmin=2)

# while curr is not None and count < limit:
#     np.append(X, curr.flatten().reshape(1000, 12288), axis=1)
#     curr = next(numpy_array_generator)
#     count += 1

# print(len(X))
# print(X.shape)

### T-SNE - LINE 104: (l, M) = np.linalg.eig(np.dot(X.T, X)), this creates an (X,X) matrix where X is the number of pixels in the image
# Y = tsne(X, 2, 50, 20.0)    # Memory error working with full size clean images, apparently even 256 x 256 BW images
# plt.scatter(Y[:, 0], Y[:, 1], 20)
# plt.show()
"""

### OCR EDA (Which OCR is the best on cropped image) ###

# OCR imports
import pytesseract

import calamari_ocr     # Only works on lines
import ocr4all

import ABBYY

# Supporting Imports
from PIL import Image
import cv2
import argparse
import pathlib
import os

"""
(Create these JSON files manually to test)
Assume JSON file with:
Image {
    Path: 'str',
    Article: [
        Polygons: [{X1: x, Y1: y, X2: x, Y2: y}]
    ]
}
"""

# Path to Images: ..\bbz-segment\05_prediction\data\pages

### construct the argument parse and parse the arguments ###
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
# 	help="type of preprocessing to be done")
# args = vars(ap.parse_args())

### TESSERACT - https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/ (Typos) ###

def use_tesseract_img(image: Image):

    # Set Tesseract Path
    pytesseract.pytesseract.tesseract_cmd = r'D:\\PyTesseract\\tesseract.exe'

    # Run OCR
    text = pytesseract.image_to_string(image)

    return text

def use_tesseract(image_path: str, preprocess: str):

    # Set Tesseract Path
    pytesseract.pytesseract.tesseract_cmd = r'D:\\PyTesseract\\tesseract.exe'

    # load the example image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print(text)

    # show the output images
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)

### Calamari OCR - (Not returning any results - https://github.com/Calamari-OCR/calamari/issues/175) ###

def use_calamari(checkpoint: str, image_path: str):
    """
        ###  ###
        checkpoint: *.ckpt.json file
    """

    # checkpoint = Path(checkpoint).resolve()
    # image_path = Path(image_path).resolve()

    ### Sub process ###
    cmd = 'calamari-predict --checkpoint \"{checkpoint}\" --files \"{image_path}\"'.format(checkpoint=checkpoint, image_path=image_path)
    subprocess.run(cmd) # Starts Calamari

### ABBYY OCR ###

def use_abbyy(image_path: str, output_path: str):

    os.environ['ABBYY_APPID'] = config['ABBYY']['AppID']
    os.environ['ABBYY_PWD'] = config['ABBYY']['AppPassword']
    os.environ['ServerUrl'] = config['ABBYY']['ServerUrl']

    pythonExecutable = 'python'
    pythonProgPath = './ABBYY/process.py'
    args = '{pythonExecutable} {pythonProgPath} {imageToProcess} {outputFile}'.format(
        pythonExecutable=pythonExecutable,
        pythonProgPath=pythonProgPath,
        imageToProcess=image_path,
        outputFile=output_path).split() 
    subprocess.call(args, shell=True) # Starts the process.py script which runs the AbbyyOnlineSDK, on a shell

### Testing OCR ###

# image_path = args["image"]
# preprocess = args["preprocess"]

# checkpoint = "D:\\Calamari\\calamari_models-1.0\\calamari_models-1.0\\antiqua_historical\\4.ckpt.json"

# output_path = './abbyy_output.txt'

# use_tesseract(image_path, preprocess)
# use_calamari(checkpoint, image_path)      # DIDNT WORK
# use_abbyy(image_path, output_path)

### Unpacking Test JSON ###

from typing import Dict
import json

output_json_filename = 'article_ocr.json'

def unpack_json():
        
    json_path = './ExampleJSONs/test_polygons.json'

    with open(json_path) as f:
        data = json.load(f)
        return data

    return {}

def image_to_article_OCR():
    """
        Input: A JSON file with the above specified variables
        Output: A JSON file of the same variables PLUS the blocks of text outputted from OCR on the bounding boxes of the images
    """
    images = unpack_json()
    is_empty = not images

    if is_empty:
        print('Empty!')
    else:

        # Create output JSON
        output_file = open(output_json_filename, "w")

        for i in range(len(images)):
            # List of images to run OCR on
            image = images[i]

            # Create "Text" key in the dictionary
            image['Text'] = []

            image_path = image['Path']
            articles = image['Article']

            # Open Image
            original_image = Image.open(image_path) # Image to extract articles from

            for j in range(len(articles)):
                # List of polygons for the Article at index "j"
                polygons = articles[j] 

                ocr_output = article_to_OCR(original_image, polygons)

                image['Text'].append(ocr_output)
    
        json.dump(images, output_file, indent=4, separators=(',', ': '))

def article_to_OCR(newspaper_image, polygons):
    """
        Input: A single article (List of bounding boxes) to run OCR on
        Output: List of text that is the OCR result from each article/bounding box
    """

    polygon_outputs = []

    for k in range(len(polygons)):
        # A single bounding box for the Article at "j" (There are multiple boxes if it spans multiple columns)
        polygon = polygons[k]

        # Should be (Left, Top, Right, Bottom)
        bounding_box = (polygon['X1'], polygon['Y1'], polygon['X2'], polygon['Y2'])

        if (not valid_crop(bounding_box)): 
            print("invalid cropping rectangle! " + str(bounding_box) + " Skipping...")
            # Have to write "Bounding Box Error" in the JSON at the index
            continue

        ### DEBUGGING ### 
        filename = get_filename(newspaper_image.filename)
        print('Article: ' + str(filename) + ", Bounding Box " + str(k) + ": " + str(bounding_box))
        #---------------#        

        # Crop image
        cropped_image = newspaper_image.crop(bounding_box)

        # Run OCR on cropped image
        ocr_ouput = use_tesseract_img(cropped_image)

        ### DEBUGGING ###
        # cropped_image.show()
        #---------------#

        ### DEBUGGING ###
        # print(ocr_ouput)
        #---------------#

        # Add ocr output into the list of outputs
        polygon_outputs.append(ocr_ouput)
    
    return polygon_outputs

def valid_crop(rectangle: Tuple[int,int,int,int]):
    is_tuple_of_four = left_top_right_bottom = False

    is_tuple_of_four = (isinstance(rectangle, tuple) and len(rectangle) == 4)
    if is_tuple_of_four: left_top_right_bottom = (rectangle[0] < rectangle[2] and rectangle[1] < rectangle[3])

    ### DEBUGGING ### 
    # print("tuple of four: " + str(is_tuple_of_four))
    # print("Correct coordinates: " + str(left_top_right_bottom))
    #---------------#

    return is_tuple_of_four and left_top_right_bottom

def path_leaf(path):
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filename(path):
    return os.path.splitext(path_leaf(path))[0]

### Testing JSON Code ###

image_to_article_OCR()
"""
Author: Alex Thomas
Creation Date: 11/29/2020
Purpose: Standalone package for running OCR on segmented articles given images and bounding coordinates
Dependencies: JSON file with:
                Image {
                    Path: 'str',
                    Article: [
                        Polygons: [{X1: x, Y1: y, X2: x, Y2: y}]
                    ]
                }
Output: JSON file with:
                Image {
                    Path: 'str'
                    Article: [
                        /* Index of text and polygon relate each set */
                        Text: [str, str, str]
                        Polygons: [{X1: x, Y1, y, X2: x,  Y2: y}]
                    ]
                }
"""

### IMPORT STATEMENTS ###
from typing import Tuple
import pytesseract
import subprocess
from PIL import Image
import argparse
import pathlib
import os
import json

### HELPER FUNCTIONS/CLASSES ###

TESSER_OCR = "tesseract"
ABBYY_OCR = "ABBYY"

abbyy_dir_path = None

output_json_filename = 'article_ocr.json'
# json_path = '../EDA/ExampleJSONs/test_polygons.json'

### OCR wrapper ###

def use_ocr(image: Image, ocr_type: str):    

    if ocr_type == TESSER_OCR:
        use_tesseract_img(image)

    # elif ocr_type == ABBYY_OCR:
    #     output_path = './results.txt'  
    #     cropped_img_name = 'temp_cropped_image.jpg' 
    #     image.save(cropped_img_name)
    #     use_abbyy(cropped_img_name, output_path)
    #     os.remove(cropped_img_name)

    else:
        print('Not a valid OCR choice')

### TESSERACT OCR ###

def use_tesseract_img(image: Image):

    # Run OCR
    text = pytesseract.image_to_string(image)

    return text

### ABBYY OCR ###

def use_abbyy(image_path: str, output_path: str):

    pythonExecutable = 'python'
    pythonProgPath = abbyy_dir_path + '/process.py'
    args = '{pythonExecutable} {pythonProgPath} {imageToProcess} {outputFile}'.format(
        pythonExecutable=pythonExecutable,
        pythonProgPath=pythonProgPath,
        imageToProcess=image_path,
        outputFile=output_path).split() 
    subprocess.call(args, shell=True) # Starts the process.py script which runs the AbbyyOnlineSDK, on a shell

    # Return results - DOESN'T WORK! Does not wait for process to be done
    results = ''

    with open(output_path, mode='r') as ocr_results:
        results = ocr_results.read()

    return results
        
### OTHER FUNCTIONS ###

def unpack_json(json_path):   

    with open(json_path) as f:
        data = json.load(f)
        return data

    return {}

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

### KEY FUNCTIONS ###

# Call this function to handle everything
def image_to_article_OCR(json_path, ocr_to_use):
    """
        Input: A JSON file with the above specified variables
        Output: A JSON file of the same variables PLUS the blocks of text outputted from OCR on the bounding boxes of the images
    """
    images = unpack_json(json_path)
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

                ocr_output = article_to_OCR(original_image, polygons, ocr_to_use)

                image['Text'].append(ocr_output)
    
        json.dump(images, output_file, indent=4, separators=(',', ': '))

def article_to_OCR(newspaper_image, polygons, ocr_to_use):
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
            polygon_outputs.append("Bounding Box Error")
            continue

        ### DEBUGGING ### 
        filename = get_filename(newspaper_image.filename)
        print('Article: ' + str(filename) + ", Bounding Box " + str(k) + ": " + str(bounding_box))
        #---------------#        

        # Crop image
        cropped_image = newspaper_image.crop(bounding_box)
        cropped_image.filename = newspaper_image.filename

        # Run OCR on cropped image
        # ocr_output = use_tesseract_img(cropped_image)
        ocr_output = use_ocr(cropped_image, ocr_to_use)

        ### DEBUGGING ###
        # cropped_image.show()
        #---------------#

        ### DEBUGGING ###
        # print(ocr_output)
        #---------------#

        # Add ocr output into the list of outputs
        polygon_outputs.append(ocr_output)
    
    return polygon_outputs

### MAIN FUNCTION (Entry Point) ###

if __name__ == '__main__':
    print('Entering Main Function...')

    ### construct the argument parse and parse the arguments ###
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json", required=True,
        help="path to input JSON file of previous step with bounding boxes for further processing")
    ap.add_argument("-t", "--tesseract", type=str, default=None,
        help="Path to the tesseract model (if you are using tesseract)")
    ap.add_argument("-a", "--abbyy", type=str, default=None,
        help="Path to the directory containing \"process.py\", \"AbbyyOnlineSdk.py\", and a config file named \"config.json\" including the ABBYY App ID, ABBYY App Password, and Server URL (if you are using ABBYY)")
    args = vars(ap.parse_args())

    ### Get JSON path of previous step to run off of
    json_path = args['json']

    ### Set Tesseract Path
    # pytesseract.pytesseract.tesseract_cmd = r'D:\\PyTesseract\\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = args['tesseract']

    ### Set ABBYY Variables
    # abbyy_dir_path = args['abbyy']

    ### If abbyy path was passed in, Initialize
    # if abbyy_dir_path:

    #     with open(abbyy_dir_path + '\\config.json') as config_json:
    #         config = json.load(config_json)
        
    #         # Loads up necessary information for ABBYY
    #         os.environ['ABBYY_APPID'] = config['ABBYY']['AppID']
    #         os.environ['ABBYY_PWD'] = config['ABBYY']['AppPassword']
    #         os.environ['ServerUrl'] = config['ABBYY']['ServerUrl']

    ### Choosing the OCR to use
    ocr_to_use = None
    if args['tesseract']:
        ocr_to_use = TESSER_OCR
    # elif args['abbyy']:
    #     ocr_to_use = ABBYY_OCR

    if ocr_to_use == None:
        print('No valid OCR choice! Exiting...')
        exit()

    # Checks for existence of JSON
    image_to_article_OCR(json_path, ocr_to_use)
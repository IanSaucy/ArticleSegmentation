"""
    Purpose: Launcher file for running the whole OCR Suite
                1. Run Model
                2. Run Bounding Box Detection Code
                3. Run OCR Code for Bounding Boxes
"""

### External Imports ###
import pytesseract
import argparse
import pathlib
import os
import json
import sys  
import importlib

### Internal Imports ###
from ImageArticleOCR.image_article_ocr import image_to_article_OCR
from extract_polygons.extract_polygons import segment_all_images

### Helper Functions/Classes ###

# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder # NOTE: Unused
def module_from_file(module_name: str, file_path: str):
    """Import a Python module with a module name, and path using ImportLib

    Args:
        module_name (str): Name of the module
        file_path (str): Path to the desired module

    Returns:
        [type]: the module you want to use - You may do module.func to use functions contained inside
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

### Importlib Imports ###

# bulk_seperator_generation_driver = module_from_file("bulk_seperator_generation_driver", "./bbz-segment/05_prediction/src/bulk_seperator_generation_driver.py")

# TODO: Refactor later into well defined modules that can be imported
sys.path.append('./bbz-segment/05_prediction/src')
from bulk_seperator_generation_driver import bulk_generate_separators

### Main Entry Point ###

def main():

    ### construct the argument parse and parse the arguments ###
    ap = argparse.ArgumentParser()
    ap.add_argument('image_directory')
    ap.add_argument('image_extensions')
    ap.add_argument('output_directory')
    ap.add_argument('model_directory')
    ap.add_argument("-r", "--regenerate", action='store_true',
        help="Re-generates the labels on images even if the files already exist")
    ap.add_argument("-d", "--debug", action='store_true',
        help="Sets debugging mode to on for more log output")
    ap.add_argument("-v", "--verbose", action='store_true',
        help="Sets to verbose mode so that it has more explanation on tasks the code is completing")
    ap.add_argument("-t", "--tesseract", type=str, default=None, required=True,
        help="Path to the tesseract executable (if you are using tesseract)")
    # ap.add_argument("-a", "--abbyy", type=str, default=None,
    #     help="Path to the directory containing \"process.py\", \"AbbyyOnlineSdk.py\", and a config file named \"config.json\" including the ABBYY App ID, ABBYY App Password, and Server URL (if you are using ABBYY)")
    args = vars(ap.parse_args())

    ### Step 1: Generate .npy file using bbz-segment and the model
    bulk_generate_separators(args['image_directory'], args['image_extensions'], args['output_directory'], args['model_directory'], args['regenerate'], args['debug'], args['verbose'])

    ### Step 2: Get bounding boxes from .npy file   
    segment_all_images(args['output_directory'], args['image_directory'], args['output_directory'], args['debug']) # TODO: When this directory and file exist uncomment this 

    ### Step 3: Run OCR on the generated bounding boxes
    JSON_NAME = 'data.json' # NOTE: This is the name of the JSON file saved at Step 2 of the pipeline
    json_path = os.path.join(args['output_directory'], JSON_NAME)
    image_to_article_OCR(json_path, args['output_directory'], args['image_directory'], "tesseract")


if __name__ == '__main__':
    main()

"""
    Purpose: Launcher file for running the whole OCR Suite
                1. Run Model
                2. Run Bounding Box Detection Code
                3. Run OCR Code for Bounding Boxes
"""

import argparse
import os
import sys

sys.path.append('./bbz-segment/05_prediction/src')
from bulk_seperator_generation_driver import bulk_generate_separators
from ImageArticleOCR.image_article_ocr import image_to_article_OCR
from extract_polygons.extract_polygons import segment_all_images

# TODO: Refactor later into well defined modules that can be imported

"""
Name of JSON file output for step 2
"""
JSON_NAME = 'data_step2.json'


def main():
    # construct the argument parse and parse the arguments ###
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
    #     help="Path to the directory containing \"process.py\", \"AbbyyOnlineSdk.py\", and a config file named
    #     \"config.json\" including the ABBYY App ID, ABBYY App Password, and Server URL (if you are using ABBYY)")
    args = vars(ap.parse_args())

    print("Running ML Model...")
    # Step 1: Generate .npy file using bbz-segment and the model
    bulk_generate_separators(args['image_directory'], args['image_extensions'], args['output_directory'],
                             args['model_directory'], args['regenerate'], args['debug'], args['verbose'])
    print("Running Image Segmentation...")
    # Step 2: Get bounding boxes from .npy file
    segment_all_images(args['output_directory'], args['image_directory'], args['output_directory'],
                       JSON_NAME, args['debug'])  # TODO: When this directory and file exist uncomment this

    print("Running OCR...")
    # Step 3: Run OCR on the generated bounding boxes
    json_path = os.path.join(args['output_directory'], JSON_NAME)
    image_to_article_OCR(json_path, args['output_directory'], args['image_directory'], "tesseract")

    print("Complete!")


if __name__ == '__main__':
    main()

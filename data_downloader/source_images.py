import csv
from os import path
from typing import Tuple, List

import requests

MAX_RETRY_COUNT = 5
ISSUE_ID_ROW = 0
IMAGE_ID_ROW = 1
IMAGE_URL_ROW = 2


def download_save_img(url: str, file_path_name: str) -> Tuple[str, bool]:
    '''
    Downloads and saves the url to the specified path. Skips files that already exist.
    Simple retry-logic for any HTTP errors controlled by global variable
    :param url: URL of resource to download
    :type url:
    :param file_path_name: Full path for where to save file, including filename
    :type file_path_name:
    :return: The path where the resource was saved and a boolean stating if it was downloaded(or skipped)
    :rtype:
    '''
    retry_count = 0
    # Check if file exists before downloading again
    if path.isfile(file_path_name):
        return file_path_name, False
    request = requests.get(url)
    while request.status_code != 200 and retry_count < MAX_RETRY_COUNT:
        request = requests.get(url)
        retry_count += 1
    if request.status_code != 200:
        raise requests.exceptions.HTTPError(request.status_code)
    with open(file_path_name, 'wb') as f:
        f.write(request.content)
    return file_path_name, True


def extract_id(full_id: str) -> str:
    '''
    Extract the ID from the full ID string.
    :param full_id:
    :type full_id:
    :return:
    :rtype:
    '''
    return full_id.split(':')[1]


def row_to_filename(row: List[str], output_path: str) -> str:
    '''
    Conversta a row to a filename. Combining both issue id and image id to form a complete filename
    :param row:
    :type row:
    :param output_path:
    :type output_path:
    :return:
    :rtype:
    '''
    raw_issue_id, raw_image_id, image_url = row[ISSUE_ID_ROW], row[IMAGE_ID_ROW], row[IMAGE_URL_ROW]
    issue_id, image_id = extract_id(raw_issue_id), extract_id(raw_image_id)
    full_path = path.join(output_path, f'{issue_id}_{image_id}.jpg')
    return full_path


def download_all_images(input_file: str, output_path: str, error_file: str) -> Tuple[int, int]:
    '''
    Downloads all images in the specified file into the specified output file. Fails gracefully and will
    attempt to continue as much as possible.
    :param input_file: The CSV file representing the URLs desired to download
    :type input_file:
    :param output_path: Directory of where to save images, must exist
    :type output_path:
    :param error_file: File name for where to save rows that have errors
    :type error_file:
    :return: (successes, failures) count
    :rtype:
    '''
    success_count = 0
    failed_count = 0
    line_count = 0
    print('\n')
    if not path.isfile(input_file):
        raise FileNotFoundError(input_file)
    if not path.isdir(output_path):
        raise FileNotFoundError(output_path)
    with open(input_file, 'r') as csv_file, open(error_file, 'w') as error_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print(f'Row: {line_count}\r', end='')
            if line_count == 0:
                line_count += 1
                continue
            output_filename = row_to_filename(row, output_path)
            try:
                _, _ = download_save_img(row[IMAGE_URL_ROW], output_filename)
                success_count += 1
            except requests.exceptions.HTTPError:
                failed_count += 1
                error_file.write(",".join(row))
                print(f'Row: {line_count} Url: {row[IMAGE_URL_ROW]} failed to download')
            except Exception as e:
                failed_count += 1
                error_file.write(",".join(row))
                print(f'Row: {line_count} Url: {row[IMAGE_URL_ROW]} failed with {e}')
            finally:
                line_count += 1
    print('\n')
    return success_count, failed_count

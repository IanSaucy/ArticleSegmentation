import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from colorama import Fore
import numpy as np
import cv2 as cv

from extract_polygons.labeled_page import LabeledPage
from extract_polygons.data_classes import Article
from extract_polygons.polygon_utils import random_color


def segment_all_images(saved_labels_dir: str, orig_img_dir: str, output_folder: str, debug=False) -> int:
    if not Path(saved_labels_dir).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(output_folder).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(orig_img_dir).exists():
        raise FileNotFoundError("Folder not found")

    numpy_file_list = list(Path(saved_labels_dir).glob('*.npy'))
    master_article_list: List[Article] = list()
    file_skip_list: List[str] = list()
    for file in tqdm(numpy_file_list):
        try:
            issue_id, image_id = str(file.stem).split('_')
        except ValueError:
            print(f'{Fore.LIGHTYELLOW_EX}\n skipped file: {str(file)}')
            print(f'could not identify issue and image ID in file name, skipping{Fore.RESET}')
            file_skip_list.append(str(file))
            continue

        loaded_data = np.load(str(file), allow_pickle=True)
        labels, original_size, filename = loaded_data.item().get('labels'), loaded_data.item().get(
            'dimensions'), loaded_data.item().get('filename')
        if labels is None or original_size is None or filename is None:
            print(f'{Fore.RED} \n failed to load labels from file {str(file)}. Skipped.')
            print(f'{Fore.RESET}')
            file_skip_list.append(str(file))
            continue
        # Must swap around, the size is given in a tuple of (height, width)
        # While the LabeledImage wants (width, height)
        original_size = original_size[1], original_size[0]
        articles = segment_single_image(labels, original_size, issue_id, image_id,
                                        src_img_path=str(Path(orig_img_dir).joinpath(filename)), debug=debug,
                                        output_folder=output_folder)
        master_article_list.extend(articles)

    # Dump data to json
    with open(Path(output_folder).joinpath('data.json'), 'w') as outfile:
        print('Opening JSON file at: ' + str(Path(output_folder).joinpath('data.json')))
        json.dump([val.JSON() for val in master_article_list], outfile)

    return len(master_article_list)


def segment_single_image(input_img_array: np.array, original_size: Tuple[int, int], issue_id: str = None,
                         img_id: str = None, src_img_path: str = None,
                         debug=False, output_folder: str = None) \
        -> List[Article]:
    if debug:
        if not Path(src_img_path).exists():
            raise FileNotFoundError
        if not Path(output_folder).exists():
            raise FileNotFoundError

    page = LabeledPage(input_img_array, original_size, img_id, issue_id, str(Path(src_img_path).name))
    # Takes a bit to run
    articles = page.segment_single_image()
    if debug:
        try:
            annotate_image(src_img_path, articles, output_folder)
        except Exception:
            print(
                f'{Fore.LIGHTYELLOW_EX}\nFailed to load source image for annotation. '
                f'\n Annotated image not saved. \n IMG: {src_img_path} {Fore.RESET}')
    return articles


def annotate_image(input_img_scr: str, articles: List[Article], output_folder: str) -> None:
    source_img = cv.imread(input_img_scr)
    source_img_name = Path(input_img_scr).stem
    for index, article in enumerate(articles):
        color = random_color()
        for box in article.img_boxes:
            # Inspired by the following:
            # https://gist.github.com/jdhao/1cb4c8f6561fbdb87859ac28a84b0201
            rect = cv.minAreaRect(box.get_contours())
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(source_img, [box], 0, color, 5)
    cv.imwrite(str(Path(output_folder).joinpath(f'{source_img_name}_annotated.jpg')), source_img)

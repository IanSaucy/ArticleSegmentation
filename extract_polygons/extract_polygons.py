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


def segment_all_images(np_file_dir: str, output_folder: str, debug=False, img_dir: str = None) -> int:
    if debug and not Path(img_dir).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(np_file_dir).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(output_folder).exists():
        raise FileNotFoundError("Folder not found")

    numpy_file_list = list(Path(np_file_dir).glob('*.npy'))
    img_src_list = defaultdict(str)
    if debug:
        # TODO: Don't static code this
        for img in Path(img_dir).glob('*.jpg'):
            img_src_list[img.stem] = str(img)
    master_article_list: List[Article] = list()
    file_skip_list: List[str] = list()
    for file in tqdm(numpy_file_list):
        try:
            issue_id, image_id = str(file.stem).split('_')
        except ValueError:
            print(f'{Fore.LIGHTYELLOW_EX}skipped file: {str(file)}')
            print(f'could not identify issue and image ID in file name, skipping{Fore.RESET}')

            file_skip_list.append(str(file))
            issue_id, image_id = None, None

        loaded_data = np.load(str(file), allow_pickle=True)
        labels, original_size = loaded_data.item().get('labels'), loaded_data.item().get('dimensions')
        print(file.stem)
        print(img_src_list.get(file.stem))
        articles = segment_single_image(labels, original_size, issue_id, image_id,
                                        input_image_src=img_src_list.get(file.stem), debug=debug,
                                        output_folder=output_folder)
        master_article_list.extend(articles)

    # Dump data to json
    with open('data.json', 'w') as outfile:
        json.dump([val.JSON() for val in master_article_list], outfile)

    return len(master_article_list)


def segment_single_image(input_img_array: np.array, original_size: Tuple[int, int], issue_id: str = None,
                         img_id: str = None, input_image_src: str = None,
                         debug=False, output_folder: str = None) \
        -> List[Article]:
    if debug:
        if not Path(input_image_src).exists():
            raise FileNotFoundError
        if not Path(output_folder).exists():
            raise FileNotFoundError

    page = LabeledPage(input_img_array, original_size, img_id, issue_id)
    # Takes a bit to run
    articles = page.segment_single_image()
    if debug:
        try:
            annotate_image(input_image_src, articles)
        except Exception:
            print(
                f'{Fore.LIGHTYELLOW_EX} Failed to load source image for annotation. '
                f'\n Annotated image not saved.{Fore.RESET}')
    return articles


def annotate_image(input_img_scr: str, articles: List[Article]) -> None:
    print(input_img_scr)
    source_img = cv.imread(input_img_scr)
    source_img_name = Path(input_img_scr).stem
    for index, article in enumerate(articles):
        color = random_color()
        for box in article.img_boxes:
            cv.rectangle(source_img, (box.top_left.col, box.top_left.row), (box.bot_right.col, box.bot_right.row),
                         color, 5)
    cv.imwrite(f'{source_img_name}_annotated.jpg', source_img)

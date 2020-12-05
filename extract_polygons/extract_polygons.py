import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
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
    issue_map = defaultdict(list)
    for file in tqdm(numpy_file_list):
        issue_id, image_id = str(file.stem).split('_')
        labels, original_size = np.load(str(file), allow_pickle=True)
        articles = segment_single_image(labels, original_size, img_dir, debug, output_folder)
        issue_map.update(issue_id, )


def segment_single_image(input_img_array: np.array, original_size: Tuple[int, int], input_image_src: str = None,
                         debug=False, output_folder: str = None) \
        -> List[Article]:
    if debug:
        if not Path(input_image_src).exists():
            raise FileNotFoundError
        if not Path(output_folder).exists():
            raise FileNotFoundError

    page = LabeledPage(input_img_array, original_size)
    # Takes a bit to run
    articles = page.segment_single_image()
    if debug:
        annotate_image(input_image_src, articles)
    return articles


def annotate_image(input_img_scr: str, articles: List[Article]) -> None:
    source_img = cv.imread(input_img_scr)
    source_img_name = Path(input_img_scr).stem
    for index, article in enumerate(articles):
        color = random_color()
        for box in article.get_boxes():
            cv.rectangle(source_img, (box.top_left.col, box.top_left.row), (box.bot_right.col, box.bot_right.row),
                         color, 5)
    cv.imwrite(f'{source_img_name}_annotated.jpg', source_img)

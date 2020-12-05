from typing import List

import numpy as np
import cv2 as cv
from Polygons.LabeledPage import LabeledPage
from Polygons.Point import Article

from extract_polygons.polygon_utils import random_color

labels = np.load('./8k71pf49w_8k71pf51x-labels.sep.npy')

page = LabeledPage(labels, (3672, 5298))
vert_seps = page.find_all_vertical_sep()
articles: List[Article] = page.find_article_boxes(vert_seps)
source_img = cv.imread('./8k71pf49w_8k71pf51x.jpg')
for index, article in enumerate(articles):
    color = random_color()
    img = cv.imread('./8k71pf49w_8k71pf51x.jpg')
    for box in article.get_boxes():
        cv.rectangle(source_img, (box.top_left.col, box.top_left.row), (box.bot_right.col, box.bot_right.row),
                     color, 5)
        cv.rectangle(img, (box.top_left.col, box.top_left.row), (box.bot_right.col, box.bot_right.row),
                     color, 5)
    cv.imwrite(f'annotated_{index}.jpg', img)
cv.imwrite(f'annotated.jpg', source_img)

# res = page._find_horz_sep_in_range(860, 1540, 240, 5279)
print(articles)
# res = page.find_all_vertical_sep()
# print(res)

# Remove everything but vertical separators
# labels_vert = reduce_to_single_label(labels, VERT)
# labels = reduce_to_single_label(labels, VERT)
# resized = cv.resize(labels_vert, (3672, 5298), interpolation=cv.INTER_AREA)

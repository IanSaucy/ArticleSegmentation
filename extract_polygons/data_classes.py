from functools import total_ordering
from typing import NamedTuple, List, Any


@total_ordering
class Point(NamedTuple):
    row: int
    col: int

    def __str__(self) -> str:
        return f'(row={self.row}, col={self.col})'

    def __lt__(self, other):
        return self.col < other.col


class ImageBox(object):
    """
    A simple class to represent a box around some region in a specific image
    """

    def __init__(self, top_left: Point, bot_right: Point, img_id: str = None, box_text: str = None):
        self._top_left = top_left
        self._bot_right = bot_right
        self._img_id = img_id
        self._box_text = box_text

    @property
    def img_id(self) -> str:
        return self._img_id

    @img_id.setter
    def img_id(self, img_id: str):
        self._img_id = img_id

    @property
    def box_text(self) -> str:
        return self._box_text

    @box_text.setter
    def box_text(self, text: str):
        self._box_text = text

    @property
    def top_left(self):
        return self._top_left

    @property
    def bot_right(self):
        return self._bot_right

    def JSON(self) -> dict:
        return {"id": self.img_id,
                "coordinates": [self.top_left.row, self.top_left.col, self.bot_right.row, self.bot_right.col],
                "text:": self.box_text}


class Article(object):
    """
    A wrapper class that contains all the boxes that designate a given article
    """

    def __init__(self, boxes: List[ImageBox], issue_id: str, title: str = None, subtitle: str = None):
        self._boxes = boxes or list()
        self._title = title
        self._subtitle = subtitle
        self._issue_id = issue_id

    def __str__(self):
        return f'{self.img_boxes}'

    def JSON(self) -> dict:
        return {"issue_id": self.issue_id, "title": self.title, "subtitle": self.subtitle,
                "images": [img.JSON() for img in self.img_boxes]}

    def add_img_box(self, box: ImageBox) -> None:
        """
        Adds a single ImageBox to the current article object
        Args:
            box (): An ImageBox

        Returns: None
        """
        self._boxes.append(box)

    @property
    def issue_id(self):
        return self._issue_id

    @issue_id.setter
    def issue_id(self, issue_id: str) -> None:
        self._issue_id = issue_id

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        self._title = title

    @property
    def subtitle(self):
        return self._subtitle

    @subtitle.setter
    def subtitle(self, subtitle: str) -> None:
        self._subtitle = subtitle

    @property
    def img_boxes(self) -> List[ImageBox]:
        return self._boxes

    @img_boxes.setter
    def img_boxes(self, boxes: List[ImageBox]):
        self._boxes = boxes


@total_ordering
class VerticalSeparator:
    """
    A simple class that represents a vertical separator
    """

    def __init__(self, top: Point, bottom: Point):
        self._top_point: Point = top
        self._bottom_point: Point = bottom

    def __str__(self) -> str:
        return f'Top: {self._top_point}, Bot: {self._bottom_point}'

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: object) -> bool:
        """
        We always sort separators based on the top point, this is useful when working on a list of separators
        """
        if isinstance(other, VerticalSeparator):
            return self._top_point < other._top_point
        else:
            return NotImplemented

    @property
    def top_point(self):
        return self._top_point

    @property
    def bottom_point(self):
        return self._bottom_point

    @top_point.setter
    def top_point(self, value):
        self._top_point = value

    @bottom_point.setter
    def bottom_point(self, value):
        self._bottom_point = value


class HorizontalSeparator:
    def __init__(self, left: Point, right: Point):
        self.left_point = left
        self.right_point = right

    def __str__(self):
        return f'Left: {self.left_point} Right: {self.right_point}'

    def __repr__(self):
        return self.__str__()

    @property
    def left_point(self):
        return self._left_point

    @property
    def right_point(self):
        return self._right_point

    @left_point.setter
    def left_point(self, value):
        self._left_point = value

    @right_point.setter
    def right_point(self, value):
        self._right_point = value

from functools import total_ordering
from typing import NamedTuple, List


@total_ordering
class Point(NamedTuple):
    row: int
    col: int

    def __str__(self) -> str:
        return f'(row={self.row}, col={self.col})'

    def __lt__(self, other):
        return self.col < other.col


class Box(NamedTuple):
    """
    A simple class to represent a box. Allows none rectangular shapes
    """
    top_left: Point
    top_right: Point
    bot_left: Point
    bot_right: Point


class Article:
    """
    A wrapper class that contains all the boxes that designate a given article
    """

    def __init__(self, boxes: List[Box]):
        self._boxes = boxes

    def __str__(self):
        return f'{self.get_boxes()}'

    def add_box(self, box: Box):
        self._boxes.append(box)

    def get_boxes(self):
        return self._boxes


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

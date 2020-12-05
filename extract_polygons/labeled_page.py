import operator
from collections import defaultdict
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np

from extract_polygons.constants import ExpectedValues
from extract_polygons.data_classes import Point, VerticalSeparator, HorizontalSeparator, Article, Box
import cv2 as cv

"""
This file contains the heart of this package. It contains all the algorithms
for interpreting a labeled image(given via a 2D array) and forming bounding boxes
around each section in the image that represent an (part) article. 

If the boxes are not being drawn correctly this is where you should look.
"""


class Labels(IntEnum):
    """
    Useful class for helping with labels in the input arrays.
    These values are what are asigned during the ML model step.
    """
    Table = 0
    Horz = 1
    Vert = 2
    Background = 3


# TODO: Break this down and reduce how much code lives in each function.
# TODO: Clear up the idea of state for this class. Should it really have state?
# TODO: Use the downsized version of the image to segment into articles \
#       then convert to original dimensions to speed up process(less pixels to scan = faster)
class LabeledPage:
    """
    A class to handle labeled pages and extracting their features
    """

    """
    The buffer we define that must exist around any given vertical separator.
    This helps to make sure we don't detect the same vertical separator multiple times
    while searching the top and bottom of the image. This value is HIGHLY dependent on the input 
    image sizes and resolutions. It also has a large affect on the performance of the search function
    """
    _separator_buffer_range: int = 150
    """
    This is the max width(really height) we expect a horizontal separator to have. 
    This value enables various parts of the program to now double count separators twice
    if they have a width/height more than 1 pixel. 
    """
    _horizontal_expected_width: int = 35

    """
    This is the number of pixel buffer that is added to the top and bottom separators when
    drawing bounding boxes for identified article regions.
    """
    _start_end_page_buffer: int = 25

    def __init__(self, img: np.array, original_size: Tuple[int, int]):
        self.img = img
        # Create single labeled copies of input image
        # Resize image while also reducing down to a single label
        self._only_vertical_labels = cv.resize(self._reduce_to_single_label(Labels.Vert), original_size,
                                               interpolation=cv.INTER_AREA)
        self._only_horizontal_labels = cv.resize(self._reduce_to_single_label(Labels.Horz), original_size,
                                                 interpolation=cv.INTER_AREA)
        self.original_size = original_size

    def segment_single_image(self) -> List[Article]:
        vertical_seps = self.find_all_vertical_sep()
        return self._find_article_boxes(vertical_seps)

    def _find_article_boxes(self, vert_sep: List[VerticalSeparator]) -> List[Article]:
        """
        A function to take a list of Vertical separators and identify the articles in each image.

        In this version of the code an article is defined as the space spanned between two horizontal
        separators across one or more columns(vertical separators). This means that an article
        could be some combination of the following:

            1. Contained within a given column
            2. Starting in one column and "flowing" into the next until it hits a separator
            3. Starting in a column and being implicitly ended by the end of the page in the last column
                the combination of both of these will force end an article even if no horizontal separator
                was detected.

        See the article class for more info, but a given article can be made up of one or more Boxes which are a series
        of (row,col) coordinates that define rectangle.

        ! Assumptions:

        This function makes several assumptions when trying to identify articles. A hopefully exhaustive list is below:
            - There are at least two vertical separators
            -

        Args:
            vert_sep (): A list of vertical separators

        Returns: A list of all identified articles

        """
        # Add a small buffer to handle text that might extend beyond the start or end of a vertical separator
        true_most_top_point, true_most_bot_point = self._find_max_vert_sep_height(vert_sep)
        buffed_most_top_point, buffed_most_bot_point = true_most_top_point - self._start_end_page_buffer, \
                                                       true_most_bot_point + self._start_end_page_buffer
        # most_bot_point = most_bot_point + self._start_end_page_buffer
        # most_top_point = most_top_point - self._start_end_page_buffer
        # Add soft separator that is the start and end of the image
        vert_sep.append(VerticalSeparator(Point(true_most_top_point, 0), Point(true_most_bot_point, 0)))
        vert_sep.append(VerticalSeparator(Point(true_most_top_point, self.original_size[0]),
                                          Point(true_most_bot_point, self.original_size[0])))
        # Super super important that the separators be sorted via columns in ascending order
        vert_sep.sort()
        img = self._only_horizontal_labels
        all_articles: List[Article] = []
        curr_article: List[Box] = []

        # Starts from the 2nd separator in the list because we define a column
        # to be the columns between two separators. Thus we need two separators!
        for index in range(1, len(vert_sep)):
            curr_sep = vert_sep[index]
            prev_sep = vert_sep[index - 1]
            # Init the top of the current box to the most top point as found previously.
            # Basically only used when starting on a new column.
            top_of_box = Point(buffed_most_top_point, prev_sep.top_point.col), Point(buffed_most_top_point,
                                                                                     curr_sep.top_point.col)
            # Sliding history of labels seen, used to avoid "double" counting horizontal separators
            sliding_history = []
            # Scan across all rows in identified range.
            # Cannot scan across the entire image since there are horizontal separators that are not related to
            # a given article. Such as the title info etc.
            for row in range(true_most_top_point, true_most_bot_point):
                if Labels.Horz in img[row,
                                  prev_sep.top_point.col:curr_sep.top_point.col] and Labels.Horz not in sliding_history:
                    # Use the current row and the column of the bottom point of the input separators
                    temp_bot_box = Point(row, prev_sep.bottom_point.col), Point(row, curr_sep.bottom_point.col)
                    temp_box = Box(top_of_box[0], top_of_box[1], temp_bot_box[0], temp_bot_box[1])
                    curr_article.append(temp_box)
                    all_articles.append(Article(curr_article))
                    # Update running top of box for the next article
                    top_of_box = temp_bot_box
                    # Clear the current article since we're starting a new article given that we've found
                    # a horizontal separator.
                    curr_article = []
                    sliding_history.append(Labels.Horz)
                elif row >= true_most_bot_point - 1:
                    # Finish off box since we're at the end of the current column within two seps
                    temp_bot_box = Point(row + self._start_end_page_buffer, prev_sep.bottom_point.col), Point(
                        row + self._start_end_page_buffer,
                        curr_sep.bottom_point.col)
                    temp_box = Box(top_of_box[0], top_of_box[1], temp_bot_box[0], temp_bot_box[1])
                    curr_article.append(temp_box)
                else:
                    sliding_history.append(-1)
                # Pruning the sliding history
                while len(sliding_history) > self._horizontal_expected_width:
                    sliding_history.pop(0)

            if index == len(vert_sep) - 1 and len(curr_article) > 0:
                # Blindly finish off this article since we're all done on this image
                temp_bot_box = Point(buffed_most_bot_point, prev_sep.bottom_point.col), Point(buffed_most_bot_point,
                                                                                              curr_sep.bottom_point.col)
                temp_box = Box(top_of_box[0], top_of_box[1], temp_bot_box[0], temp_bot_box[1])
                curr_article.append(temp_box)
                all_articles.append(Article(curr_article))
        return all_articles

    def _find_max_vert_sep_height(self, vert_sep: List[VerticalSeparator]) -> Tuple[int, int]:
        """
        Takes a list of vertical separators and finds the the (row,col) point for both extremes, both "highest" and "lowest"
        "highest" and "lowest" is a bit of a confusing term here given that a separator that is "higher" physically on an image
        will have a lower row number since coordinates start at the top left.

        Args:
            vert_sep (): A list of vertical separators

        Returns: A tuple representing the two extrema values.

        """
        most_top_sep_cord = vert_sep[0].top_point.row
        most_bot_sep_cord = vert_sep[0].bottom_point.row
        for sep in vert_sep:
            if sep.top_point.row < most_top_sep_cord:
                most_top_sep_cord = sep.top_point.row
            if sep.bottom_point.row > most_bot_sep_cord:
                most_bot_sep_cord = sep.bottom_point.row
        return most_top_sep_cord, most_bot_sep_cord

    def _find_horz_sep_in_range(self, start_col: int, end_col: int, start_row: int, end_row: int) \
            -> List[Optional[HorizontalSeparator]]:
        """
        ! Not Used but might be helpful

        Finds all the horizontal separators in the specified range, a combination of col and row.
        Searches across the image data stored on the instance of the object, scanning over the filtered
        image that only contains horizontal image labels.

        It defines a separator as ANY pixel(some [row,col] value) containing the horizontal separator label.
        In addition, this separator will span the entire input range of columns.

        It attempts to not "double count" separators that have a width greater than a single pixel(most) by
        keeping a history and not counting a found separator if there is still a separator in the history.
        Args:
            start_col (): The column where the search should start, also the start of any separators that are created
            end_col (): The column where the search should end, also the end of any separators that are created
            start_row (): What row to start searching on(inclusive)
            end_row (): What row to stop searching on (exclusive)

        Returns:
            A list of HorizontalSeparators representing the identified horizontal separators
        """
        if self._only_vertical_labels.shape[1] < end_col:
            raise IndexError
        if start_col < 0:
            raise IndexError
        img = self.only_horizontal_labels
        horz_sep_list: List[HorizontalSeparator] = []
        label_history: List[int] = []
        for row in range(start_row, end_row):
            # Checks if within the range there is a horizontal label. In addition, and most importantly
            # there is not a label in the history -- this makes sure we don't "double count" separators
            # since they're of a width greater than 1 pixel.
            if Labels.Horz in img[row, start_col:end_col] and Labels.Horz not in label_history:
                # TODO: Maybe it should be more restrictive than just a single pixel being defined as a separator
                # We just define the separator as the first row that contains a horizontal separator pixel
                # which means it's possible there is some separator below this designation(not a problem)
                horz_sep_list.append(HorizontalSeparator(Point(row, start_col), Point(row, end_col)))
                label_history.append(Labels.Horz)
            else:
                # This value does not matter much. This is just handy to spot bugs
                label_history.append(-1)
            # This makes sure we don't input the same separator multiple times since it has a width greater
            # a single pixel. See note above
            if len(label_history) > self._horizontal_expected_width:
                label_history.pop(0)

        return horz_sep_list

    def find_all_vertical_sep(self) -> List[VerticalSeparator]:
        """
        A function to find all the vertical separators in the image, attempting to find the true start of each(top and bottom of image)

        Once the top and bottom of vertical separators are found it attempts to match them up by identifying a given
        bottom and top separator that are within a specified range of columns from each other.

        Verifies that the number of top and bottom points(representing the start and end of some separators) are equal
        and that the final value of separators is the same as the number of separator points previously identified.


        """
        # TODO This really should be broken down into some smaller bits
        # List of points representing the top and bottom of identified separators.
        # ! Slow function call!
        top_of_separators: List[Point] = self._find_top_of_vert_sep()
        bot_of_separators: List[Point] = self._find_bot_of_vert_sep()
        # Make sure both have the same number of points
        assert len(top_of_separators) == len(bot_of_separators)
        # Sort to make it easier to match up the bottom and top points of a given separator
        top_of_separators.sort()
        bot_of_separators.sort()
        complete_vertical_separator: List[VerticalSeparator] = []
        for top_point in top_of_separators:
            for bot_point in bot_of_separators:
                # TODO: It would be much better to remove elems from the array as we go.
                # But given that we know that they're going to be small, less than whatever our max number of columns is
                # we can ignore this and just do an O(n^2) search.
                if bot_point.col - self._separator_buffer_range < top_point.col < bot_point.col + self._separator_buffer_range:
                    # These are a match
                    complete_vertical_separator.append(VerticalSeparator(top_point, bot_point))
        assert len(complete_vertical_separator) == len(bot_of_separators)
        complete_vertical_separator.sort()
        return complete_vertical_separator

    def _find_bot_of_vert_sep(self) -> List[Point]:
        """
        Searches the vertical image data stored on this object for the "top" of all vertical separators.

        Always tries to identify the top most pixel that makes up some separator. Whatever the top most
        pixel is will be used to define the start point of a given separator. In other words, the width of the
        separator is not factored into the data returned.

        A buffer is set around any given separator, this buffer prevents double counting a given separator
        due to its potential width. This also means that there is a minimum distance that any two separators must have
        to have detected as two separate separators in place of a single separator.

        Returns: A list of Points that represent the top of all identified separators

        """
        # Handy local variable
        img = self.only_vertical_labels
        # height, width = np.shape - Just a reminder
        # Set a max number of rows to be searched. In other words, don't search beyond this row
        top_search_limit: int = img.shape[0] // 3
        expected_number_seps: int = self._find_number_vert_sep(self._only_vertical_labels)
        number_found_seps: int = 0
        # This acts as our "buffer keeper" anything in this set is already part of some other
        # separator.
        found_separator_ranges = set()
        found_separators: List[Point] = []
        for row in reversed(range(top_search_limit, img.shape[0])):
            if number_found_seps == expected_number_seps:
                # stop searching once we found the number of separators we expected to see.
                break
            # Scan over all columns of the input image
            for col in range(img.shape[1]):
                if col in found_separator_ranges:
                    # This is a separator we've already seen
                    # TODO: Come up with a better way to skip ranges that are already part of a separator
                    continue
                if img[row, col] == Labels.Vert:
                    # This is a separator!
                    found_separators.append(Point(row, col))
                    number_found_seps += 1
                    # Add range to what we've already seen so we won't accidentally double add it to our list
                    found_separator_ranges.update(
                        [x for x in range(col - self._separator_buffer_range, col + self._separator_buffer_range)])
                    break
        return found_separators

    def _find_top_of_vert_sep(self) -> List[Point]:
        """
        Searches the vertical image data stored on this object for the "top" of all vertical separators.

        ! See the related _find_bot_of_vert_sep() for more detailed comments

        Always tries to identify the top most pixel that makes up some separator. Whatever the top most
        pixel is will be used to define the start point of a given separator. In other words, the width of the
        separator is not factored into the data returned.

        A buffer is set around any given separator, this buffer prevents double counting a given separator
        due to its potential width. This also means that there is a minimum distance that any two separators must have
        to have detected as two separate separators in place of a single separator.

        Returns: A list of Points that represent the top of all identified separators

        """
        img = self.only_vertical_labels
        bottom_search_limit: int = img.shape[0] // 2
        expected_number_seps: int = self._find_number_vert_sep(self._only_vertical_labels)
        number_found_seps: int = 0
        found_separator_ranges = set()
        found_separators: List[Point] = []
        for row in range(bottom_search_limit):
            if number_found_seps == expected_number_seps:
                break
            for col in range(img.shape[1]):
                if col in found_separator_ranges:
                    # This is a separator we've already seen
                    continue
                if img[row, col] == Labels.Vert:
                    # This is a separator!
                    found_separators.append(Point(row, col))
                    number_found_seps += 1
                    # Add range to what we've already seen so we won't accidentally double add it to our list
                    found_separator_ranges.update(
                        [x for x in range(col - self._separator_buffer_range, col + self._separator_buffer_range)])
                    break
        return found_separators

    def _reduce_to_single_label(self, desired_label: Labels) -> np.array:
        """
        Takes the input image and removes all labels except the specified one.
        ! Warning: Always replaces labels to be removed with 0(and there is a label which itself is 0)
        Args:
            desired_label (): Label desired to be kept

        Returns: The modified numpy array

        """
        unique = np.unique(self.img)
        temp_img = np.copy(self.img)
        for label in unique:
            if label != desired_label:
                temp_img[temp_img == label] = 0
        return temp_img

    def _find_number_vert_sep(self, input_image: np.array) -> int:
        """
        Takes the image and finds the number of vertical separators. Searches multiple
        places in the image to ideally come up with the "true" number of vertical separators.
        The number found is checked against a sane max number expected value as a rudimentary
        sanity check.
        Args:
            input_image: (): Input image in the form of a numpy array containing only
            vertical separator labels.

        Returns: The number of vertical separators found

        """
        start_loc = input_image.shape[0] // 2
        step_size = int(input_image.shape[0] * 0.2)
        locations_to_test = [start_loc - step_size, start_loc, start_loc + step_size]

        vertical_sep_locs = []
        vertical_sep_counts = defaultdict(int)
        sliding_window_history = []
        for row in locations_to_test:
            for col, value in enumerate(input_image[row]):
                if value == Labels.Vert and Labels.Vert not in sliding_window_history:
                    vertical_sep_locs.append(Point(row, col))
                    vertical_sep_counts[row] += 1
                sliding_window_history.append(value)
                if len(sliding_window_history) > 3:
                    sliding_window_history.pop(0)

        # Find the row that has the most number of vertical separators
        max_vert_sep_row = max(vertical_sep_counts.items(), key=operator.itemgetter(1))[0]
        assert vertical_sep_counts[max_vert_sep_row] < ExpectedValues.maxNumberOfVerticalSeparators
        return vertical_sep_counts[max_vert_sep_row]

    @property
    def only_vertical_labels(self):
        return self._only_vertical_labels

    @property
    def only_horizontal_labels(self):
        return self._only_horizontal_labels

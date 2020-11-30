"""
Author: Alex Thomas
Creation Date: 11/29/2020
Purpose: Standalone package for running OCR on segmented articles given images and bounding coordinates
Dependencies: JSON file with:
                Image {
                    Path: 'str',
                    Article: [
                        Polygons: [{X1: x, Y1: y, X2: x, Y2: y}]
                    ]
                }
Output: JSON file with:
                Image {
                    Path: 'str'
                    Article: [
                        /* Index of text and polygon relate each set */
                        Text: [str, str, str]
                        Polygons: [{X1: x, Y1, y, X2: x,  Y2: y}]
                    ]
                }
"""

### HELPER FUNCTIONS/CLASSES ###


def image_to_article_OCR():
    """
        Input: A JSON file with the above specified variables
        Output: A JSON file of the same variables PLUS the blocks of text outputted from OCR on the bounding boxes of the images
    """
    pass

def article_to_OCR():
    """
        Input: A single article (List of bounding boxes) to run OCR on
        Output: List of text that is the OCR result from each article/bounding box
    """
    pass

### MAIN FUNCTION (Entry Point) ###

if __name__ == '__main__':
    print('Entering Main Function...')
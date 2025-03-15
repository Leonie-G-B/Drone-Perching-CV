
# General utility functions that can be called from any script

import cv2
import os
import time
from functools import wraps 
import numpy as np
import matplotlib.pyplot as plt


### Misc util funcs ###


def timeit():
    stats = {"count": 0, "total_time": 0}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time
            stats["count"] += 1
            stats["total_time"] += duration

            print(f"\nFunction '{func.__name__}' Call #{stats['count']}: {duration:.6f}s")
            print(f"Total Execution Time: {stats['total_time']:.6f}s\n")

            return result
        return wrapper
    return decorator


def check_contains(tuple, str: str)-> bool:
    return any([t in str for t in tuple])

## Img processing utils ##


def load_in_img(img_path: str) -> 'np.ndarray':
    """
    Load in an image from a file path.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Image data.
    """
    assert os.path.exists(img_path), f"Image file {img_path} not found."

    img = cv2.imread(img_path)
    return img


def mask_img_to_binary(image: np.ndarray) -> np.ndarray:
    """
    Takes the input segmented image and converts it into the correct form for the skeletonisation algorithm"""

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    # binary_image = binary_image.astype(bool)
    
    return binary_image

def shifter(l, n):
    return l[n:] + l[:n]

def distance(x, x1, y, y1):
    return np.sqrt((x - x1) ** 2.0 + (y - y1) ** 2.0)

def is_tpoint(vallist):
    '''
    Determine if point is part of a block:
    [X X X]
    [0 X 0]
    And all 90 deg rotation of this shape

    If there are only 3 connections, this is an end point. If there are 4,
    it is a body point, and if there are 5, it remains an intersection

    '''

    vals = np.array(vallist)

    if vals.sum() < 3:
        return False

    arrangements = [np.array([0, 6, 7]), np.array([0, 1, 2]),
                    np.array([2, 3, 4]), np.array([4, 5, 6])]

    posns = np.where(vals)[0]

    # Check if all 3 in an arrangement are within the vallist
    for arrange in arrangements:
        if np.in1d(posns, arrange).sum() == 3:
            return True

    return False


def is_blockpoint(vallist):
    '''
    Determine if point is part of a block:
    [X X]
    [X X]

    Will have 3 connected sides, with one as an 8-connection.
    '''

    vals = np.array(vallist)

    if vals.sum() < 3:
        return False

    arrangements = [np.array([0, 1, 7]), np.array([1, 2, 3]),
                    np.array([3, 4, 5]), np.array([5, 6, 7])]

    posns = np.where(vals)[0]

    # Check if all 3 in an arrangement are within the vallist
    for arrange in arrangements:
        if np.in1d(posns, arrange).sum() == 3:
            return True

    return False


## Img/ results visualisation utils ##


def visualise_result(input, category: str, img_shape : tuple = None) -> None: 
    """
    Call at any point in the pipeline to visualise the current state. State given by 'category'.
    Note that there is no mechanism by which to ensure the user has given the correct category (this method will likely just fail in that case).
    Current supported list of inputs: 
    'img' - original image (np.ndarray)
    'segmentation' - the output of the segmentation model (np.ndarray)
    'medial_axis' - the output of the medial axis calculation (np.ndarray)
    'branches' - list of arrays containing branch points
    """

    fig, ax = plt.subplots(figsize = (8, 8))

    

    if category == 'img': # TESTED
        ax.imshow(input)
    elif category == 'segmentation': # TESTED
        ax.set_title("Segmentation result")
        ax.imshow(input, cmap = 'gray')
    elif category == 'medial_axis': # TESTED 
        med_skel = ax.imshow(input, cmap= 'hot', alpha = 0.8)
        cbar = fig.colorbar(med_skel, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Local skeleton width (pixels)', fontsize = 12)
        ax.set_title("Medial axis result")
        ax.set_xticks([])
        ax.set_yticks([])
    elif category == 'branches': # TESTED
        for branch in input: 
            ax.plot(branch[:, 1], img_shape[0] - branch[:, 0])
    else: 
        print("Invalid category given. \nOptions: 'img', 'segmentation', 'medial_axis', 'branches")

    return 
    
        




# General utility functions that can be called from any script

import cv2
import os
import time
from functools import wraps 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import itertools
import random
import networkx as nx

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




def load_in_rdom_imgs(img_dir: str, mask_dir: str) -> np.ndarray:

    os.listdir(img_dir)

    img_loc = os.path.join(img_dir, random.choice(os.listdir(img_dir)))

    # mask is in the form .png
    mask_loc = os.path.join(mask_dir, os.path.basename(img_loc).replace(".jpg",".png"))

    assert os.path.exists(img_loc), f"Image file {img_loc} not found."
    assert os.path.exists(mask_loc), f"Mask file {mask_loc} not found."

    return img_loc, mask_loc


## Img processing utils ##


def sliding_window(array: list[tuple], window_size: int, stride: float = 1):
    """
    Yields points along the array with the specified window size and stride (number of pixels separating each consecutive window).
    """
    for i in range(0, len(array) - window_size + 1, stride):
        yield array[i:i + window_size]


def grad(array: list[tuple]) -> tuple:

    x = np.array([i[0] for i in array])
    y = np.array([i[1] for i in array])

    if len(np.unique(x)) == 1:
        return 90

    # first, fit a linear approximation: 
    m , _ = np.polyfit(x, y, 1) # m = gradient, second unused variable is +c

    # determine local angle
    # first_deriv = np.gradient(y, x) 
    angle_rad = np.arctan(m) #  reverse tan of grad = local angle
    angle_deg = np.degrees(angle_rad)

    return angle_deg
    

def curvature(array: list[tuple], window_size: int = 25) -> np.ndarray:

    x = np.array([i[0] for i in array])
    y = np.array([i[1] for i in array])

    # Compute first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Compute curvature using the formula
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    curvature_smoothed = np.convolve(curvature, np.ones(window_size) / window_size, mode='same')
    curvature_smoothed = np.nan_to_num(curvature_smoothed, nan=0)

    return curvature_smoothed


def widths(array: list, pixel_to_length_ratio) -> tuple[float, float, float]:

    np_list = np.array(array) * pixel_to_length_ratio
    average_width = np.mean(array)* pixel_to_length_ratio 

    return average_width, np_list.min(), np_list.max()


def get_branch_width(branch_pixels, medial_axis) -> np.ndarray:

    branch_width = []

    for branch_pixel in branch_pixels: 
        # identify the width at this point 
        width = medial_axis[branch_pixel[0][1], branch_pixel[0][0]]
        assert width, "No width found at branch point"
        branch_width.append(width)  

    return branch_width


def decide_branch_weighting(branch_widths: np.ndarray, max_width : float, max_length: float ,length: float,
                            pixel_to_length_ratio: float,
                            width_ideal_threshold: tuple = (30, 110),
                            length_to_width_bias : float = 0.6) -> float:
    
    a = length_to_width_bias

    width_threshold_pixels = (pixel_to_length_ratio * width_ideal_threshold[0], pixel_to_length_ratio * width_ideal_threshold[1])

    weighting = 0

    # proportion of branch that is within threshold 
    in_threshold = sum(width_threshold_pixels[0] <= w <=width_threshold_pixels[1] 
                       for w in branch_widths)
    total        = len(branch_widths)
    in_threshold_proportion = in_threshold/total

    average_width =  sum(branch_widths) / total

    weighting = a* (length/max_length) + (1-a) *(in_threshold_proportion * (average_width/max_width))

    return weighting

# @timeit()
def determine_branch_label(branch: np.ndarray, labels: np.ndarray) -> int:
    """
    Determine the label of the branch given the labels in an image like representation of the skeleton.
    """
    
    # Use a point at the midpoint of the branch to ensure we aren't
    # at a point in the label where the intersection has been removed

    midpoint_idx = len(branch) // 2
    midpoint = branch[midpoint_idx]
    if len(midpoint) ==1: 
        midpoint = midpoint[0]

    label = labels[midpoint[1], midpoint[0]]
    if not label: 
        print("WARNING: NO LABEL FOUND")

    return label



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


def product_gen(n):
    for r in itertools.count(1):
        for i in itertools.product(n, repeat=r):
            yield "".join(i)


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


def merge_nodes(node, G):
    '''
    Combine a node into its neighbors.
    '''

    neigb = list(G[node])

    if len(neigb) != 2:
        return G

    G.remove_node(node)
    G.add_edge(neigb[0], neigb[1])

    return G


## Img/ results visualisation utils ##



def get_rectangle_patch(x, y, angle, width) -> patches.Rectangle:

    x_start, x_end = x[0], x[-1]
    y_start, y_end = y[0], y[-1]

    midpoint_id = len(x) // 2
    midpoint = (x[midpoint_id], y[midpoint_id])

    # get length from start to end (x and y)
    length = distance(x_start, x_end, y_start, y_end)

    rect = patches.Rectangle(
        (midpoint[0] - length / 2, midpoint[1] - width / 2),  # Bottom-left corner before rotation
        length, width,  # Width and height
        angle=angle,  # Rotation angle in degrees
        edgecolor='r',
        facecolor='none',
        linewidth=2
    )

    return rect



def visualise_result(input, category: str, 
                     img_shape : tuple = None, underlay_img : bool = False, img: np.ndarray = None, 
                     nodes: dict = None, highlight_ids: list = None, annotations: dict = None,
                     title : str = None) -> None: 
    """
    Call at any point in the pipeline to visualise the current state. State given by 'category'.
    Note that there is no mechanism by which to ensure the user has given the correct category (this method will likely just fail in that case).
    Current supported list of inputs: 
    'img' - original image (np.ndarray).
    'segmentation' - the output of the segmentation model (np.ndarray).
    'medial_axis' - the output of the medial axis calculation (np.ndarray)
    'branches' - list of arrays containing branch points.
    'branch_weightings' - dictionary containing branch information (inluding all branch pixels and their weightings).
    'nodes' - list of nodes.
    'graph' - networkx graph object. Include node dict for labelling and positioning.
    'highlight' - plots branches with specified branches plotted in red to highlight them.
    """

    fig, ax = plt.subplots(figsize = (8, 8))


    if category == 'img': # TESTED
        image_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB) 
        ax.imshow(image_rgb)
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
        if underlay_img: 
            assert img is not None, "No image given to underlay"
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            ax.imshow(image_rgb, alpha = 0.6, origin = 'upper')

        if type(input) == dict:
            for branch in input.values():
                banch_pixels = branch[0]
                ax.plot(banch_pixels[:, 0][:, 0], img_shape[0] - banch_pixels[:, 0][:, 1])
        else:
            try: 
                for branch in input: 
                    ax.plot(branch[:, 1], img_shape[0] - branch[:, 0])
            except IndexError:
                for branch in input: 
                    ax.plot(branch[:,0][:, 0], img_shape[0] - branch[:,0][:, 1])
    elif category == 'branch_weightings':
        cmap = plt.get_cmap('viridis')
        norm_colours = plt.Normalize(vmin = 0, vmax = 1)
        # the input is the branch_properties variable

        if underlay_img: 
            assert img is not None, "No image given to underlay"
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            ax.imshow(image_rgb, alpha = 0.6, origin = 'upper')

        for branch_label in input: 
            pixels = input[branch_label][0]
            weighting = input[branch_label][2]
            colour = cmap(norm_colours(weighting))
            if underlay_img:
                ax.plot(pixels[:, 0][:,0], pixels[:, 0][:, 1], color = colour)
            else: 
                ax.plot( pixels[:, 0][:, 0], img_shape[0] -pixels[:, 0][:, 1], color = colour )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_colours)
        plt.colorbar(sm, ax = ax, label="Branch Weight")
        plt.title("Sectioned tree branches' calculated weightings.")
        plt.axis('off')
    elif category == 'nodes': 
        if underlay_img: 
            assert img is not None, "No image given to underlay"
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            ax.imshow(image_rgb, alpha = 0.6, origin = 'upper')
        for node in input: 
            ax.scatter(node[0], img_shape[0] - node[1], 'ro')
    elif category == 'graph':
        if nodes: 
            nx.draw(input , pos = nodes, with_labels = True)
        else: 
            nx.draw(input)
    elif category == 'highlight':
        if highlight_ids is None:
            print("No specific branches to highlight provided.") 
            return
        if underlay_img: 
            assert img is not None, "No image given to underlay"
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            ax.imshow(image_rgb, alpha = 0.6, origin = 'upper')

        for branch_label in input: 
            x = input[branch_label][0][:,0][:,0]
            if underlay_img:
                y = input[branch_label][0][:,0][:,1]
            else:
                y = img_shape[0] - input[branch_label][0][:,0][:,1]

            if branch_label in highlight_ids:
                ax.plot(x, y, color = 'red')
                # ax.annotate(f"Label: {branch_label}", xy = (x[len(x)//2], y[len(y)//2]))
            else:
                ax.plot(x, y, color = 'black', linewidth = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    else: 
        print("Invalid category given. \nOptions: 'img', 'segmentation', 'medial_axis', 'branches, 'branch_weightings', 'nodes', 'graph'")
        return

    if title: 
        ax.set_title(title)

    return 
    
        


def visualise_metrics(full_input, metrics: dict, img: np.ndarray = None):

    fig, ax = plt.subplots(figsize = (8, 8))


    # visualise the whole tree, and then plot the one specific spotlight branch(es)
    # with its attriubutes labelled
    
    # Plot the original image (brackground, faded)
    if img is not None: 
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        ax.imshow(image_rgb, alpha = 0.6, origin = 'upper')

    # Plot the full_input
    for branch_label in full_input: 
        x = full_input[branch_label][0][:,0][:,0]
        y = full_input[branch_label][0][:,0][:,1]
        ax.plot(x, y)
        try:
            annotate_info = metrics[branch_label]
        except KeyError:
            continue
        else: 
            midpoint = (x[len(x)//2], y[len(y)//2])
            ax.annotate(f"Angle: {annotate_info['angle']}", 
                        xy = midpoint)


    # Annotate the metrics 

    pass



def visualise_curvature(full_input, curvatures, window_size: int = 25):

    _ , ax = plt.subplots(figsize = (8, 8))

    # cmap = plt.get_cmap('viridis') 
    cmap = plt.get_cmap('coolwarm') 
    all_curvatures = np.concatenate(list(curvatures.values()))
    print(np.min(all_curvatures))
    print(np.max(all_curvatures))
    norm_colours = plt.Normalize(vmin=np.min(all_curvatures), vmax=np.max(all_curvatures)) # Normalise by largest and smallest curvature value

    for branch_label in full_input:
        if branch_label not in curvatures:
            continue

        x = full_input[branch_label][0][:,0][:,0]
        y = - full_input[branch_label][0][:,0][:,1]
        curvature = curvatures[branch_label]

        num_segments = len(x) // window_size
        curvature = np.asarray(curvature)

        for i in range(num_segments):
            start_idx = i * window_size
            end_idx = start_idx + window_size

            if end_idx >= len(x):  # Avoid going out of bounds
                break

            # Extract the segment
            x_segment = x[start_idx:end_idx]
            y_segment = y[start_idx:end_idx]

            # Pick the curvature value from the start of the segment (or midpoint)
            curvature_value = curvature[start_idx]

            # Map curvature to color
            color = cmap(norm_colours(curvature_value))

            # Plot the segment
            ax.plot(x_segment, y_segment, color=color, linewidth=2)

        
        # x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
        # y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

        # # segments = [[(x1, y1), (x2, y2)], ...]
        # segments = np.array([
        #     [(x_midpts[i], y_midpts[i]), (x_midpts[i+1], y_midpts[i+1])]
        #     for i in range(len(x_midpts)-1)
        # ])

        # lc = LineCollection(segments, cmap=cmap, norm=norm_colours, linewidths=2)
        # lc.set_array(curvature[:-1]) 
        # ax.add_collection(lc)  


    # ax.autoscale()  
    # ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_colours)
    sm.set_array([])  # Required for colorbar
    plt.colorbar(sm, ax=ax, label="Curvature")

    ax.set_xticks([])
    ax.set_yticks([])

    # plt.show()

    return
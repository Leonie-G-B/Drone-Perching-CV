
# IMPORTS 


import cv2 
import numpy as np 
import os 
from typing import Literal
import warnings

from utils import timeit


# FUNCTIONS (secondary / smaller)


def is_clockwise(contour):
    """Check if a contour is orientated clockwise."""
    value = sum((p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]) for p1, p2 in zip(contour, contour[1:] + contour[:1]))
    return value < 0


## Explanation for a 'clockwise contour':
# Clockwise contour:
## If you walk along the contour in order, the enclosed area in on your right.

# Anti-clockwise contour:
## If you walk along the contour in order, the enclosed area is on your left.


def get_merge_point_idx(contour1, contour2):
    """Find the closest points between two contours."""
    idx1, idx2 = 0, 0
    min_distance = float('inf')
    if len(contour1) > 10000: 
        warnings.warn(f"Contour 1 is large ({len(contour1)})- processing time may be extensive.")
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = (p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2
            if distance < min_distance:
                min_distance = distance
                idx1, idx2 = i, j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    """Merge parent and child contours at the closest points."""
    contour = np.concatenate((contour1[:idx1+1], contour2[idx2:], contour2[:idx2+1], contour1[idx1:]), axis=0)
    return contour

def merge_with_parent(parent_contour, child_contour):
    """Ensure correct contour orientation and merge child into parent."""
    if not is_clockwise(parent_contour):
        parent_contour = parent_contour[::-1]
    if is_clockwise(child_contour):
        child_contour = child_contour[::-1]
    idx1, idx2 = get_merge_point_idx(parent_contour, child_contour)
    return merge_contours(parent_contour, child_contour, idx1, idx2)

def group_child_contours_with_parent(hierarchy):
    """Organize contours into parent-child relationships."""
    groups = {}
    for i, h in enumerate(hierarchy.squeeze()):
        parent_idx = h[3]
        if parent_idx != -1:
            groups.setdefault(parent_idx, {"parent": parent_idx, "child": []})["child"].append(i)
        else:
            groups.setdefault(i, {"parent": i, "child": []})
    return groups


def list_mask_paths_n_rdom(dir: str, n : int = 1) -> list[str]:
    """
    For the number, n, given, return a list of n pasks to mask files located in the specified directory, dir.
    """
    mask_files = os.listdir(dir)
    import random

    indices = random.sample(range(0, len(mask_files)), n)

    return [os.path.join(dir, mask_files[i]) for i in indices]

def list_mask_paths(dir: str) -> list[str]: #maybe move this to utilies if something similar is needed for another script..
    """
    List all paths (filename and path included) in a str list.
    """
    mask_files = os.listdir(dir)
    return [os.path.join(dir, mask_file) for mask_file in mask_files]


# FUNCTIONS (main / larger)

@timeit()
def convert_mask_to_yolo_seg_label(mask_path, output_txt_path, visualise=False, resize_factor: float =  0.2):
    """Convert a binary mask image to YOLO segmentation format."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
    height, width = mask_resized.shape

    _, thresh = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or hierarchy is None:
        return None

    test_mask = np.zeros((height, width), dtype=np.uint8)  # Visual verification mask

    label_str = ""
    if hierarchy.shape[1] > 1:  # Handle nested contours
        contour_groups = group_child_contours_with_parent(hierarchy)
        for group in contour_groups.values():
            parent_contour = contours[group["parent"]]
            for child_idx in group["child"]:
                parent_contour = merge_with_parent(parent_contour, contours[child_idx])

            contour_to_write = parent_contour.squeeze()
            if len(contour_to_write) < 3:
                continue  # Skip invalid contours

            for point in contour_to_write:
                label_str += f" {round(float(point[0]) / width, 6)} {round(float(point[1]) / height, 6)}"
            label_str = f"0{label_str}\n"  # Assume class ID = 0

            cv2.drawContours(test_mask, [parent_contour], -1, 255, -1)
    else:
        contour_to_write = contours[0].squeeze()
        if len(contour_to_write) < 3:
            return None

        for point in contour_to_write:
            label_str += f" {round(float(point[0]) / width, 6)} {round(float(point[1]) / height, 6)}"
        label_str = f"0{label_str}\n"

        cv2.drawContours(test_mask, [contours[0]], -1, 255, -1)

    with open(output_txt_path, "w") as f:
        f.write(label_str.strip())

    if visualise:
        cv2.imshow("Original Mask", mask_resized)
        cv2.imshow("Reconstructed Mask", test_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return test_mask

def convert_yolo_label_to_mask(image_shape, label_file):
    """Convert YOLO txt file back into a binary mask image."""
    height, width = image_shape
    mask = np.zeros((height, width), np.uint8)

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            x_points = list(map(lambda x: int(float(x) * width), parts[1::2]))
            y_points = list(map(lambda y: int(float(y) * height), parts[2::2]))

            pts = np.array(list(zip(x_points, y_points)), np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

    return mask

def process_masks(mask_list, output_folder="yolo/dataset/labels", 
                  visualise_img : bool = False, 
                  resize_factor : float = 0.2,
                  dir : Literal['train', 'val'] = 'train',
                  save_imgs     : bool = False,
                  keep_previous_progress : bool = True):
    """Process a list of masks, convert them to YOLO format, validate them visually, and continue on keypress if visualising."""

    output_folder = os.path.join(output_folder, dir)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for mask_path in mask_list:
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        output_txt = os.path.join(output_folder, f"{base_name}.txt")
        if os.path.exists(output_txt) and keep_previous_progress:
            print(f"Skipping {mask_path} (already processed)")
            continue
        output_visual = os.path.join(output_folder, f"{base_name}_reconstructed.png")

        print(f"Processing: {mask_path}")

        # Convert mask to YOLO format
        test_mask = convert_mask_to_yolo_seg_label(mask_path, output_txt, visualise = visualise_img, resize_factor=resize_factor)

        if test_mask is None:
            print(f"Skipping {mask_path} (no valid contours found)")
            continue

        if visualise_img:  
            # Convert YOLO back to mask
            reconstructed_mask = convert_yolo_label_to_mask(test_mask.shape, output_txt)
            

            # Display original and reconstructed mask for validation
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            original_mask = cv2.resize(original_mask, (0, 0), fx=resize_factor, fy=resize_factor)
            combined = np.hstack([original_mask, reconstructed_mask])
            cv2.imshow(f"Validation - {mask_path}", combined)

            print(f"YOLO label saved: {output_txt}")
            print(f"Reconstructed mask saved: {output_visual}")
            print("Press any key to continue to the next mask...")

            cv2.waitKey(0)  # Wait for user to validate before moving on
            cv2.destroyAllWindows()

        if save_imgs:
            cv2.imwrite(output_visual, reconstructed_mask)


def main(mask_dir: str,
         resize_factor : float = 0.2,
         dir : Literal['train', 'val', 'both'] = 'both'):
    """
    Main function to process the masks into the .txt yolo format.
    This is setup for one time use, though some small modifications allow use for testing and visualisation for mask validation.
    For example, by calling list_mask_paths_n_rdom, you can select a random number of masks to process instead of all at once, 
    and process_masks has an additional optional variable, visualise, which will show the original mask and processed one for reference.

    Args:
        mask_dir (str): The directory containing the masks to be processed. 
                        Expected format ends with /yolo/dataset/masks, and then the final dir is determined by the dir argument.
        resize_factor (float): The factor by which the masks are resized. If not resized, each process will take >20s!
        dir (Literal['train', 'val', 'both']): The directory to take from. If both, then functions will be processed twice in succession. 
                                                Options are 'train', 'val', or 'both'. 
                                                Default is 'both'.
    """

    dirs_to_process = ['train', 'val'] if dir == 'both' else [dir]

    for sub_dir in dirs_to_process:
        current_mask_dir = os.path.join(mask_dir, sub_dir)
        
        if not os.path.exists(current_mask_dir):
            raise FileNotFoundError(f"Directory not found: {current_mask_dir}")
        
        if not os.listdir(current_mask_dir):
            raise FileNotFoundError(f"Directory is empty: {current_mask_dir}")

        
        mask_list = list_mask_paths(current_mask_dir)
        
        process_masks(mask_list, resize_factor=resize_factor, dir=sub_dir)

        print(f"Processing Complete for {sub_dir} masks.")
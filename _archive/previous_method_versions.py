

import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage.morphology import skeletonize





@utils.timeit()
def order_branch_pixels(branch_coords: np.ndarray) -> np.ndarray:
    """
    Order the branch pixels to create a line plot that should follow the original skeleton. 
    """
    # branch_coords = list(branch_coords) #pop only works if its a list
    # ordered_pixels = [branch_coords.pop(0)]

    branch_coords = {tuple(p) for p in branch_coords}  # Convert to set of tuples
    ordered_pixels = [branch_coords.pop()]  # Pop an arbitrary starting pixel

    while branch_coords:
        last_pixel = ordered_pixels[-1]
        # find the closest pixel in the remaining list
        # next_pixel = min(branch_coords, key=lambda p: distance.euclidean(last_pixel.tolist(), p.tolist()))
        next_pixel = min(branch_coords, key=lambda p: distance.euclidean(last_pixel, p))
        ordered_pixels.append(next_pixel)
        branch_coords.remove(next_pixel)

    return np.array(ordered_pixels)


@utils.timeit()
def order_branch_pixels_KD(branch_coords: np.ndarray, max_step: int = 5) -> np.ndarray:
    branch_coords = np.array(branch_coords)  # Ensure it's a numpy array
    tree = KDTree(branch_coords)  # Build KD-Tree
    ordered_pixels = []
    skipped_points = []


    # Start from the first point
    current_idx = 0
    visited = set([current_idx])
    ordered_pixels.append(branch_coords[current_idx])

    for _ in range(len(branch_coords) - 1):
        distances, nearest_indices = tree.query(branch_coords[current_idx], k=len(branch_coords))  # Get all neighbors
        distances, counts = np.unique(distances, return_counts=True)
        duplicates = distances[counts > 1]
        
        min_distance = np.min(distances[distances>0])
        if any(duplicates == min_distance):
            # multiple minimum distances, append them both!
            pass
        
        for dist, idx in zip(distances, nearest_indices):
            if idx not in visited and dist <= max_step:  # Only accept close & unvisited neighbors
                current_idx = idx
                visited.add(idx)
                ordered_pixels.append(branch_coords[idx])
                break
            elif (idx not in visited) and (dist > max_step):
                # hmmmmm
                pass
        else:
            break  # If no valid next point is found, stop early


    return np.array(ordered_pixels)

@utils.timeit()
def order_branch_pixels_KD2(branch_coords: np.ndarray, max_step: int = 5) -> np.ndarray:

    branch_coords = np.array(branch_coords)  
    branch_coords = branch_coords.astype(np.float64)
    tree = KDTree(branch_coords, leafsize=50) 
    ordered_pixels = []

    current_idx = 0
    visited = set([current_idx])
    ordered_pixels.append(branch_coords[current_idx])

    if branch_coords[0,0] == 662:
        print("stop here and step through")

    

    for _ in range(len(branch_coords) - 1):
        
        # print(current_idx)
        re = False
        distances, nearest_indices = tree.query(branch_coords[current_idx], k=len(branch_coords))  # Get all neighbors
        distances_unique, counts_unique = np.unique(distances, return_counts=True)
        duplicates = distances_unique[counts_unique > 1]  # Get the distances that have duplicates
        
        # Find the smallest non-zero distance (ensuring we don't take 0 as valid distance)
        min_distance = np.min(distances[distances > 0])

        if any(duplicates == min_distance):
            # Find all indices corresponding to the smallest duplicate distances
            duplicate_indices = nearest_indices[distances == min_distance]
            for idx in duplicate_indices:
                if idx not in visited:
                    current_idx = idx
                    ordered_pixels.append(branch_coords[idx])
                    visited.add(idx)
                    re = True
                    break

        if re: 
            continue

        # Proceed to the nearest unvisited point within the max_step constraint
        for dist, idx in zip(distances, nearest_indices):
            if idx not in visited and dist <= max_step:  # Only accept close & unvisited neighbors
                current_idx = idx
                visited.add(idx)
                ordered_pixels.append(branch_coords[idx])
                break
            elif (idx not in visited) and (dist > max_step):
                # Handle skipped points or any other logic for far distances
                print("something here?")
                pass
        else:
            break  # If no valid next point is found, stop early

    if len(ordered_pixels) < len(branch_coords):
        print("Warning: Not all points were visited.")

    return np.array(ordered_pixels)


def order_branch_pixeles_NN(branch_coords: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    branch_coords = np.array(branch_coords)  
    branch_coords = branch_coords.astype(np.float64)

    nn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="euclidean")
    nn.fit(branch_coords)

    ordered_pixels = []

    current_idx = 0
    visited = set([current_idx])
    ordered_pixels.append(branch_coords[current_idx])

    distances, nearest_indices = nn.kneighbors([branch_coords[current_idx]])

    for _ in range(len(branch_coords) - 1):
        
        # print(current_idx)
        re = False
        distances, nearest_indices = tree.query(branch_coords[current_idx], k=len(branch_coords))  # Get all neighbors
        distances_unique, counts_unique = np.unique(distances, return_counts=True)
        duplicates = distances_unique[counts_unique > 1]  # Get the distances that have duplicates
        
        # Find the smallest non-zero distance (ensuring we don't take 0 as valid distance)
        min_distance = np.min(distances[distances > 0])

        if any(duplicates == min_distance):
            # Find all indices corresponding to the smallest duplicate distances
            duplicate_indices = nearest_indices[distances == min_distance]
            for idx in duplicate_indices:
                if idx not in visited:
                    current_idx = idx
                    ordered_pixels.append(branch_coords[idx])
                    visited.add(idx)
                    re = True
                    break

        if re: 
            continue

        # Proceed to the nearest unvisited point within the max_step constraint
        for dist, idx in zip(distances, nearest_indices):
            if idx not in visited and dist <= max_step:  # Only accept close & unvisited neighbors
                current_idx = idx
                visited.add(idx)
                ordered_pixels.append(branch_coords[idx])
                break
            elif (idx not in visited) and (dist > max_step):
                # Handle skipped points or any other logic for far distances
                print("something here?")
                pass
        else:
            break  # If no valid next point is found, stop early

    if len(ordered_pixels) < len(branch_coords):
        print("Warning: Not all points were visited.")

    return np.array(ordered_pixels)
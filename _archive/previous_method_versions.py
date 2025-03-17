

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




def create_graph(labels, branches, branch_labels, intersects, end_pts) -> nx.Graph:
    """
    Converts skeletonized structures into a graph representation.
    """

    G = nx.Graph()

    end_nodes = {}
    inter_nodes = []
    nodes = []
    branch_lengths = {}
    
    for branch_num, branch in enumerate(branches): 
        branch_lengths[branch_labels[branch_num]] = cv2.arcLength(branch, False)

    # END POINTS
    for end in end_pts:
        # identify the branch label
        branch_label = labels[labels.shape[0] - end[1] -1, end[0]]
        end_nodes[branch_label] = end
        # all edge points are named with the convention E_{branch_label}

    # Create end point nodes
    for end_point_label in end_nodes:
        G.add_node(f"E_{end_point_label}_{end_nodes[end_point_label]}", 
                    label = end_point_label, 
                    type = 'endpoint', 
                    position = end_nodes[end_point_label], 
                    length= branch_lengths[end_point_label]) 

    
    # INTERSEC POINTS
    # As the intersections have been removed from the labeled array, 
    # we need to have a 8-connected search radius for the relevant branch label
    intersec_labels = {}
    intersec_set_temp = set() #use a set so we don't have duplicate entries 

    for intersec in intersects:
        if len(intersec) >9: 
            print("pause here")
            # intersec = intersec[0]
        if isinstance(intersec, tuple): # i.e there is only one point at the intersection
            intersec = [intersec] # otherwise we are just iterating through the x and y point instead of it as a set

        intersec_set_temp.clear()

        for i in intersec:
            # for dx, dy in [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            #                 (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            #                 ( 0, -2), ( 0, -1),          ( 0, 1), ( 0, 2),
            #                 ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
            #                 ( 2, -2), ( 2, -1), ( 2, 0), ( 2, 1), ( 2, 2)]:
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                           ( 0, -1),          ( 0, 1),
                           ( 1, -1), ( 1, 0), ( 1, 1)]:
                neighbour_x = i[0] + dx
                neighbour_y = i[1] + dy
                if 0 <= neighbour_x < labels.shape[1] and 0 <= neighbour_y < labels.shape[0]:
                    branch_label = labels[neighbour_x, neighbour_y]
                    if branch_label > 0:
                        intersec_set_temp.add(branch_label)
            # assert len(intersec_set_temp) > 0, "No branch label found for intersection point. NEEDS DEBUGGING."

        first_pixel = intersec[0]

        for found_branch in intersec_set_temp:
            if found_branch in intersec_labels:
                intersec_labels[found_branch].append(first_pixel) 
            else:
                intersec_labels[found_branch] = [first_pixel]


    # Create intersection nodes
    for intersec_branch_label in intersec_labels:
        for intersec in intersec_labels[intersec_branch_label]:
            G.add_node(f"I_{intersec_branch_label}_{intersec}", 
                       label = intersec_branch_label,
                       type = 'intersection', 
                       position = intersec,
                       length = branch_lengths[intersec_branch_label])

    # Now add edges
    # start by locating the end points and then iterating along
    # the length of the branch, adding the edges from end point
    # to intersection, and intersection to intersection (if any)

    edge_list = set()

    # Edge connections
    edge_list_temp = []
    loops_temp = []

    for branch_label, end in end_nodes.items():
        if branch_label in intersec_labels:
            for intersec in intersec_labels[branch_label]:
                edge = (f"E_{branch_label}_{end}", f"I_{branch_label}_{intersec}", branch_label)
                edge_list_temp.append(edge)

    # Connecting intersection nodes
    inter_nodes = {branch_label: [(f"I_{branch_label}_{intersec}", intersec_labels[branch_label])] 
                   for branch_label in intersec_labels}

    for branch_label, inter_list in inter_nodes.items():
        for i, inters in enumerate(inter_list):
            for j, inters_2 in enumerate(inter_list):
                if i != j:
                    match = list(set(inters[1]) & set(inters_2[1]))  # Find common connections

                    if len(match) == 1:
                        new_edge = (inters[0], inters_2[0], match[0])
                    elif len(match) > 1:
                        # Pick the best connection (shortest length or best intensity)
                        multi = [branch_lengths[m[0]] for m in match]
                        keep = multi.index(min(multi))
                        new_edge = (inters[0], inters_2[0], match[keep])

                        for jj in range(len(multi)):
                            if jj == keep:
                                continue
                            loop_edge = (inters[0], inters_2[0], match[jj])
                            if loop_edge not in loops_temp and (loop_edge[1], loop_edge[0], loop_edge[2]) not in loops_temp:
                                loops_temp.append(loop_edge)

                    if new_edge is not None:
                        if (new_edge[1], new_edge[0], new_edge[2]) not in edge_list_temp and new_edge not in edge_list_temp:
                            edge_list_temp.append(new_edge)

    for edge in edge_list_temp:
        G.add_edge(edge[0], edge[1])



    for end_node_label, end_point in end_nodes.items():
        branch_pix = branches[branch_labels.index(end_node_label)]
        equiv_pixs = [branch_pix[:,0][:,0], labels.shape[0] - 1 - branch_pix[:,0][:,1]]
        first_point = (equiv_pixs[0][0], equiv_pixs[1][0])
        last_point = (equiv_pixs[0][-1], equiv_pixs[1][-1])

        # identify if the edge is at the start or end of the branch
        if not end_point == first_point:
            # reverse the branch
            equiv_pixs = [equiv_pixs[0][::-1], equiv_pixs[1][::-1]]
            first_point = (equiv_pixs[0][0], equiv_pixs[1][0])
            last_point = (equiv_pixs[0][-1], equiv_pixs[1][-1])
        
        # Now check for intersection points along this branch 

        for intersec_points in intersec_labels[end_node_label]:
            if len(intersec_points) == 1: 
                # perfect - just add the edge point between the end and intersection
                G.add_edge(f"E_{end_node_label}_{end_point}", f"I_{end_node_label}_{intersec_points}")
            else: 
                # must go through the pixels of the branch and find the correct order of the edge points! 
                pass

    for end_label, end_points in end_nodes.items():
        if end_label in intersec_labels:
            if len(intersec_labels[end_label]) == 1: 
                G.add_edge(f"E_{end_label}_{end_points}", f"I_{end_label}_{intersec_labels[end_label]}")

    for end_label, end_points in end_nodes.items():
        if end_label in intersec_labels:  # Only connect if the end_label exists in intersection labels
                for intersec_point in intersec_labels[end_label]:
                    # Add edge between the end point and intersection point
                    G.add_edge(f"E_{end_label}_{end_points}", f"I_{end_label}_{intersec_point}")

    return edge_list, nodes, inter_nodes

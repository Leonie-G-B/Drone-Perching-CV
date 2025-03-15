
# IMPORTS

import utils 

# image processing packages
from skimage.morphology import medial_axis # , skeletonize
import scipy.ndimage as nd
from scipy.spatial import distance, KDTree
import cv2

# other useful packages
import numpy as np


# This is the location of every processing step post segmentation,
# excluding the final algorithm, ranking, and results.


# Define the class: tree that will be populated with all relevant information as processing occurs

# Class definition where appropriate datafields for the tree are populated
# and 

class Tree: 
    def __init__(self, img: np.ndarray, segmentation, resize: float = 1.0):
        """
        Main object: the tree and all associated information required for the decision making algorithm. 

        Args:
            img (np.ndarray): The input image of the tree.
            segmentation (np.ndarray): The output of the segmentation model. 
            resize (float): The factor by which to resize the input image and segmentation output. Default = 1.0 (i.e. no change to original size).
        """
        self.img = cv2.resize(img, None, fx = resize, fy = resize) # Input image
        self.segmentation = cv2.resize(segmentation, None, fx = resize, fy = resize) # The output of the segmentation model
        
        self.binary_mask = utils.mask_img_to_binary(segmentation) # Binary mask required for medial axis.
        self.skeleton = None
        self.branches = None

    def populate_medial_axis(self, verbose: bool = False):
        self.skeleton, self.medial_axis = perf_medial_axis(self.binary_mask)
        print("Completed medial axis calc")
        if verbose:
            print("Plotting medial axis result.")
            utils.visualise_result(self.medial_axis, 'medial_axis')

    @utils.timeit()
    def section_branches(self):
        
        labels, num = nd.label(self.skeleton, np.ones((3, 3)))

        if num>1: 
            print('More than one object found... may cause untested behaviour.') # potential issue - keep an eye on this 
        
        # from this point onwards, going to assume there is only one skeleton
        # can be changed - need to assess the implications of there being multiple skeletons

        skel_points, intersec_pts, skeleton_no_intersec, endpts = find_filpix(1, labels, final=True, debug=False, remove_region= True)
        # skeleton_no_intersec is necessary to separate the skeleton into branches
        
        branch_labels, num_branches = nd.label(skeleton_no_intersec, np.ones((3, 3)))
        # branch_coords = {i: np.argwhere(branch_labels == i) for i in range(1, num_branches + 1)}
        branch_coords = [np.argwhere(branch_labels == i) for i in range(1, num_branches + 1)]

        utils.visualise_result(branch_coords, 'branches', img_shape = self.img.shape)
        # At this point the coordinates are separate for each branch but theyre in the wrong order...
        ordered_branch_coords = []
        for branch in branch_coords:
            # first quickly determine the curvature of the branch
            # if curvature indicates there is overlap in the x or y direction, we need to order the points properly

            ordered_branch_coords.append(order_branch_pixels_KD2(branch))
        
        utils.visualise_result(ordered_branch_coords, 'branches', img_shape = self.img.shape)

        print("Done")


@utils.timeit()
def perf_medial_axis(binary_mask: np.ndarray) -> np.ndarray:
    
    skeleton, distance = medial_axis(binary_mask, return_distance=True)
    distance_on_skeleton = distance * skeleton
    return skeleton, distance_on_skeleton


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



@utils.timeit()
def find_filpix(branches, labelfil, final=True, debug=False, remove_region : bool = False):
    '''
    MODIFIED SLIGHTLY FROM THE FILFINDER PACKAGE. See: https://github.com/e-koch/FilFinder.

    Identifies the types of pixels in the given skeletons. Identification is
    based on the connectivity of the pixel.

    Parameters
    ----------
    branches : list
        Contains the number of branches in each skeleton. 
        Should be 1 in this use case.
    labelfil : list
        Contains the skeleton array.
    final : bool, optional
        If true, corner points, intersections, and body points are all
        labeled as a body point for use when the skeletons have already
        been cleaned.
    debug : bool, optional
        Enable to print out (a lot) of extra info on pixel classification.

    Returns
    -------
    fila_pts : list
        All points on the body of each skeleton.
    inters : list
        All points associated with an intersection in each skeleton.
    labelfil : list
       Contains the arrays of each skeleton where all intersections
       have been removed.
    endpts_return : list
        The end points of each branch of each skeleton.
  '''

    initslices = []
    initlist = []
    shiftlist = []
    sublist = []
    endpts = []
    blockpts = []
    bodypts = []
    slices = []
    vallist = []
    shiftvallist = []
    cornerpts = []
    subvallist = []
    subslist = []
    pix = []
    filpix = []
    intertemps = []
    fila_pts = []
    inters = []
    repeat = []
    temp_group = []
    all_pts = []
    endpts_return = []

    for k in range(1, branches + 1):
        x, y = np.where(labelfil == k)
        for i in range(len(x)):
            if x[i] < labelfil.shape[0] - 1 and y[i] < labelfil.shape[1] - 1:
                pix.append((y[i], labelfil.shape[0] - x[i] - 1))
                initslices.append(np.array([[labelfil[x[i] - 1, y[i] + 1],
                                             labelfil[x[i], y[i] + 1],
                                             labelfil[x[i] + 1, y[i] + 1]],
                                            [labelfil[x[i] - 1, y[i]], 0,
                                                labelfil[x[i] + 1, y[i]]],
                                            [labelfil[x[i] - 1, y[i] - 1],
                                             labelfil[x[i], y[i] - 1],
                                             labelfil[x[i] + 1, y[i] - 1]]]))

        filpix.append(pix)
        slices.append(initslices)
        initslices = []
        pix = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            initlist.append([slices[i][k][0, 0],
                             slices[i][k][0, 1],
                             slices[i][k][0, 2],
                             slices[i][k][1, 2],
                             slices[i][k][2, 2],
                             slices[i][k][2, 1],
                             slices[i][k][2, 0],
                             slices[i][k][1, 0]])
        vallist.append(initlist)
        initlist = []

    for i in range(len(slices)):
        for k in range(len(slices[i])):
            shiftlist.append(utils.shifter(vallist[i][k], 1))
        shiftvallist.append(shiftlist)
        shiftlist = []

    for k in range(len(slices)):
        for i in range(len(vallist[k])):
            for j in range(8):
                sublist.append(
                    int(vallist[k][i][j]) - int(shiftvallist[k][i][j]))
            subslist.append(sublist)
            sublist = []
        subvallist.append(subslist)
        subslist = []

    # x represents the subtracted list (step-ups) and y is the values of the
    # surrounding pixels. The categories of pixels are ENDPTS (x<=1),
    # BODYPTS (x=2,y=2),CORNERPTS (x=2,y=3),BLOCKPTS (x=3,y>=4), and
    # INTERPTS (x>=3).
    # A cornerpt is [*,0,0] (*s) associated with an intersection,
    # but their exclusion from
    #   [1,*,0] the intersection keeps eight-connectivity, they are included
    #   [0,1,0] intersections for this reason.
    # A blockpt is  [1,0,1] They are typically found in a group of four,
    # where all four
    #   [0,*,*] constitute a single intersection.
    #   [1,*,*]
    # A T-pt has the same connectivity as a block point, but with two 8-conns
    # [*, *, *]
    # [0, 1, 0]
    # The "final" designation is used when finding the final branch lengths.
    # At this point, blockpts and cornerpts should be eliminated.
    for k in range(branches):
        for l in range(len(filpix[k])):
            x = [j for j, y in enumerate(subvallist[k][l]) if y == k + 1]
            y = [j for j, z in enumerate(vallist[k][l]) if z == k + 1]

            if len(x) <= 1:
                if debug:
                    print("End pt. {}".format(filpix[k][l]))
                endpts.append(filpix[k][l])
                endpts_return.append(filpix[k][l])
            elif len(x) == 2:
                if final:
                    bodypts.append(filpix[k][l])
                else:
                    if len(y) == 2:
                        if debug:
                            print("Body pt. {}".format(filpix[k][l]))
                        bodypts.append(filpix[k][l])

                    elif utils.is_tpoint(vallist[k][l]):
                        # If there are only 3 connections to the t-point, it
                        # is an end point
                        if len(y) == 3:
                            if debug:
                                print("T-point end {}".format(filpix[k][l]))
                            endpts.append(filpix[k][l])
                            endpts_return.append(filpix[k][l])
                        # If there are 4, it is a body point
                        elif len(y) == 4:
                            if debug:
                                print("T-point body {}".format(filpix[k][l]))

                            bodypts.append(filpix[k][l])
                        # Otherwise it is a part of an intersection
                        else:
                            if debug:
                                print("T-point inter {}".format(filpix[k][l]))
                            intertemps.append(filpix[k][l])
                    elif utils.is_blockpoint(vallist[k][l]):
                        if debug:
                            print("Block pt. {}".format(filpix[k][l]))
                        blockpts.append(filpix[k][l])
                    else:
                        if debug:
                            print("Corner pt. {}".format(filpix[k][l]))
                        cornerpts.append(filpix[k][l])
            elif len(x) >= 3:
                if debug:
                    print("Inter pt. {}".format(filpix[k][l]))
                intertemps.append(filpix[k][l])
        endpts = list(set(endpts))
        bodypts = list(set(bodypts))
        dups = set(endpts) & set(bodypts)
        if len(dups) > 0:
            for i in dups:
                bodypts.remove(i)
        # Cornerpts without a partner diagonally attached can be included as a
        # bodypt.
        if debug:
            print("Cornerpts: {}".format(cornerpts))

        if len(cornerpts) > 0:
            deleted_cornerpts = []

            for i, j in zip(cornerpts, cornerpts):
                if i != j:
                    if utils.distance(i[0], j[0], i[1], j[1]) == np.sqrt(2.0):
                        proximity = [(i[0], i[1] - 1),
                                     (i[0], i[1] + 1),
                                     (i[0] - 1, i[1]),
                                     (i[0] + 1, i[1]),
                                     (i[0] - 1, i[1] + 1),
                                     (i[0] + 1, i[1] + 1),
                                     (i[0] - 1, i[1] - 1),
                                     (i[0] + 1, i[1] - 1)]
                        match = set(intertemps) & set(proximity)
                        if len(match) == 1:
                            print("MATCH")
                            bodypts.extend([i, j])
                            deleted_cornerpts.append(i)
                            deleted_cornerpts.append(j)
            cornerpts = list(set(cornerpts).difference(set(deleted_cornerpts)))

        if len(cornerpts) > 0:
            for l in cornerpts:
                proximity = [(l[0], l[1] - 1),
                             (l[0], l[1] + 1),
                             (l[0] - 1, l[1]),
                             (l[0] + 1, l[1]),
                             (l[0] - 1, l[1] + 1),
                             (l[0] + 1, l[1] + 1),
                             (l[0] - 1, l[1] - 1),
                             (l[0] + 1, l[1] - 1)]

                # Check if the matching corner point is an end point
                # Otherwise the pixel will be combined into a 2-pixel intersec
                match_ends = set(endpts) & set(proximity[-4:])
                if len(match_ends) == 1:
                    fila_pts.append(endpts + bodypts + [l])
                    continue

                match = set(intertemps) & set(proximity)
                if len(match) == 1:
                    intertemps.append(l)
                    fila_pts.append(endpts + bodypts)
                else:
                    fila_pts.append(endpts + bodypts + [l])
        else:
            fila_pts.append(endpts + bodypts)

        # Reset lists
        cornerpts = []
        endpts = []
        bodypts = []

        
        if len(blockpts) > 0:
            for i in blockpts:
                all_pts.append(i)
        if len(intertemps) > 0:
            for i in intertemps:
                all_pts.append(i)
        # Pairs of cornerpts, blockpts, and interpts are combined into an
        # array. If there is eight connectivity between them, they are labelled
        # as a single intersection.
        arr = np.zeros((labelfil.shape))
        for z in all_pts:
            # print(z) ; print(labelfil[z[1]]); print(labelfil[labelfil.shape[0] - z[1] -1, z[0]])
            row, col = labelfil.shape[0] - z[1] - 1 , z[0]
            if remove_region:
                # remove a 3x3 pixel region around the intersection point
                r_min , r_max = max(0, row - 1), min(labelfil.shape[0], row + 2)
                c_min , c_max = max(0, col - 1), min(labelfil.shape[1], col + 2)
                labelfil[r_min:r_max, c_min:c_max] = 0
            else:
                labelfil[row, col] = 0
                arr[z[1], z[0]] = 1
        lab, nums = nd.label(arr, np.ones((3,3)))
        for k in range(1, nums + 1):
            objs_pix = np.where(lab == k)
            for l in range(len(objs_pix[0])):
                temp_group.append((objs_pix[0][l], objs_pix[1][l]))
            inters.append(temp_group)
            temp_group = []
    for i in range(len(inters) - 1):
        if inters[i] == inters[i + 1]:
            repeat.append(inters[i])
    for i in repeat:
        inters.remove(i)

    if debug:
        print("Fila pts: {}".format(fila_pts))
        print("Intersections: {}".format(inters))
        print("End pts: {}".format(endpts_return))
        print(labelfil)

    return fila_pts, inters, labelfil, endpts_return

# IMPORTS

import utils 

# image processing packages
from skimage.morphology import medial_axis # , skeletonize
import scipy.ndimage as nd
from scipy.spatial import distance, KDTree
import cv2
import networkx as nx

# other useful packages
import numpy as np
import string

import matplotlib.pyplot as plt # useful during debugging

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

        skel_points, intersec_pts, skeleton_no_intersec, endpts = find_filpix(1, labels, final=True, debug=False, remove_region= False)
        # skeleton_no_intersec is necessary to separate the skeleton into branches
        
        labeltree, num_branches = nd.label(skeleton_no_intersec, np.ones((3, 3)))
        # branch_coords = {i: np.argwhere(branch_labels == i) for i in range(1, num_branches + 1)}
        binary_skel_no_intersec  = (skeleton_no_intersec > 0).astype(np.uint8) * 255
        # branches, _ = cv2.findContours(binary_skel_no_intersec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        branches, _ = cv2.findContours(binary_skel_no_intersec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #no chain approx otherwise we lose some pixels from each line!
        
        branch_labels = []
        for branch in branches: 
            branch_labels.append(utils.determine_branch_label(branch, labeltree))


        # utils.visualise_result(branches, 'branches', img_shape = self.img.shape)
        branch_properties = {}
        for branch, branch_lablel in zip(branches, branch_labels): 
            branch_properties[branch_lablel] = [branch, cv2.arcLength(branch, False)]

        pre_graph(labeltree, branch_properties, intersec_pts, endpts)

        create_graph(labeltree, branches, branch_labels, intersec_pts, endpts)

        print("Done")


@utils.timeit()
def perf_medial_axis(binary_mask: np.ndarray) -> np.ndarray:
    
    skeleton, distance = medial_axis(binary_mask, return_distance=True)
    distance_on_skeleton = distance * skeleton
    return skeleton, distance_on_skeleton




def pre_graph(labelisofil, branches, interpts, ends):
    '''

    This function converts the skeletons into a graph object compatible with
    networkx. The graphs have nodes corresponding to end and
    intersection points and edges defining the connectivity as the branches
    with the weights set to the branch length.

    CONTINUE

    '''


    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []
    loop_edges = []

    lengths = []
    

    # def path_weighting(idx, length, intensity, w=0.5):
    #     '''

    #     Relative weighting for the shortest path algorithm using the branch
    #     lengths and the average intensity along the branch.

    #     '''
    #     if w > 1.0 or w < 0.0:
    #         raise ValueError(
    #             "Relative weighting w must be between 0.0 and 1.0.")
    #     return (1 - w) * (length[idx] / np.sum(length)) + \
    #         w * (intensity[idx] / np.sum(intensity))

    # lengths = branch_properties["length"]
    # branch_intensity = branch_properties["intensity"]

    inter_nodes_temp = []
    # Create end_nodes, which contains lengths, and nodes, which we will
    # later add in the intersections
    end_nodes.append([(labelisofil[labelisofil.shape[0] - end[1] -1, end[0]],
                       branches[labelisofil[labelisofil.shape[0] - end[1] -1, end[0]]][1]) # length
                      for end in ends])
                    #    path_weighting(int(labelisofil[n][i[0], i[1]] - 1))) #,
                    #                   lengths[n],
                    #                   branch_intensity[n]),
                    #    lengths[n][int(labelisofil[n][i[0], i[1]] - 1)],,

                    #    branch_intensity[n][int(labelisofil[n][i[0], i[1]] - 1)])
                    #   for i in ends[n]])
    # nodes.append([labelisofil[i[0], i[1]] for i in ends])

# Intersection nodes are given by the intersections points of the filament.
# They are labeled alphabetically (if len(interpts[n])>26,
# subsequent labels are AA,AB,...).
# The branch labels attached to each intersection are included for future
# use.
    inter_nodes_temp = []

    for intersec in interpts:  # assuming interpts is iterable
        for i in intersec:  
            uniqs = {
                (labelisofil[i[0] + dx, i[1] + dy],  # branch label
                branches[labelisofil[i[0] + dx, i[1] + dy]][1])  # length
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                            ( 0, -1),          ( 0, 1),
                            ( 1, -1), ( 1, 0), ( 1, 1)]
                if 0 <= i[0] + dx < labelisofil.shape[1] and 0 <= i[1] + dy < labelisofil.shape[0]
                and labelisofil[i[0] + dx, i[1] + dy] > 0
            }
            inter_nodes_temp.append(list(uniqs))  

    # Add intersection labels
    inter_nodes.append(list(zip(utils.product_gen(string.ascii_uppercase), inter_nodes_temp)))

    for alpha, node in zip(utils.product_gen(string.ascii_uppercase), inter_nodes_temp):
        nodes.append(alpha)


    for intersec in interpts:
        
        for i in intersec:  
            uniqs = {
                labelisofil[i[0] + dx, i[1] + dy]
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                            ( 0, -1),          ( 0, 1),
                            ( 1, -1), ( 1, 0), ( 1, 1)]
                if 0 <= i[0] + dx < labelisofil.shape[1] and 0 <= i[1] + dy < labelisofil.shape[0]
                and labelisofil[i[0] + dx, i[1] + dy] > 0
            }
            inter_nodes_temp.append(list(set(uniqs)))  

    inter_nodes.append(list(zip(utils.product_gen(string.ascii_uppercase), inter_nodes_temp)))
    
    for alpha, node in zip(utils.product_gen(string.ascii_uppercase), inter_nodes_temp):
        nodes.append(alpha)
          
    
    # Edges are created from the information contained in the nodes.
    edge_list_temp = []
    loops_temp = []
    for i, inters in enumerate(inter_nodes):
        end_match = list(set(inters[1]) & set(end_nodes))
        for k in end_match:
            edge_list_temp.append((inters[0], k[0], k))

        for j, inters_2 in enumerate(inter_nodes):
            if i != j:
                match = list(set(inters[1]) & set(inters_2[1]))
                new_edge = None
                if len(match) == 1:
                    new_edge = (inters[0], inters_2[0], match[0])
                elif len(match) > 1:
                    # Multiple connections (a loop)
                    multi = [match[l][1] for l in range(len(match))]
                    keep = multi.index(min(multi))
                    new_edge = (inters[0], inters_2[0], match[keep])

                    # Keep the other edges information in another list
                    for jj in range(len(multi)):
                        if jj == keep:
                            continue
                        loop_edge = (inters[0], inters_2[0], match[jj])
                        dup_check = loop_edge not in loops_temp and \
                            (loop_edge[1], loop_edge[0], loop_edge[2]) \
                            not in loops_temp
                        if dup_check:
                            loops_temp.append(loop_edge)

                if new_edge is not None:
                    dup_check = (new_edge[1], new_edge[0], new_edge[2]) \
                        not in edge_list_temp \
                        and new_edge not in edge_list_temp
                    if dup_check:
                        edge_list_temp.append(new_edge)

    # Remove duplicated edges between intersections

    edge_list.append(edge_list_temp)
    loop_edges.append(loops_temp)

    return edge_list, nodes, loop_edges



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

    # Add edges to the graph
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
                arr[labelfil.shape[0] - z[1] - 1, z[0]] = 1
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
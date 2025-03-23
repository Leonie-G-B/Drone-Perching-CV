
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
import operator
import math

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
    def section_branches(self, pixel_to_length_ratio: float = None, verbose: bool = False):


        labels, num = nd.label(self.skeleton, np.ones((3, 3)))

        if num>1: 
            # this is the case where there are multiple trees in one image
            # for now we will just use the first one
            print("Multiple trees detected. Using the first one.")
            for row in labels: 
                for item in row: 
                    if item>1: 
                        item == 1
        
        # from this point onwards, going to assume there is only one skeleton
        # can be changed - need to assess the implications of there being multiple skeletons

        skel_points, intersec_pts, skeleton_no_intersec, endpts = find_branchpix(1, labels, final=True, debug=False, remove_region= False)
        # skeleton_no_intersec is necessary to separate the skeleton into branches
        
        labeltree, num_branches = nd.label(skeleton_no_intersec, np.ones((3, 3)))

        # branch_coords = {i: np.argwhere(branch_labels == i) for i in range(1, num_branches + 1)}
        binary_skel_no_intersec  = (skeleton_no_intersec > 0).astype(np.uint8) * 255
        # branches, _ = cv2.findContours(binary_skel_no_intersec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        branches, _ = cv2.findContours(binary_skel_no_intersec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #no chain approx otherwise we lose some pixels from each line!
        
        branch_labels = []
        for branch in branches: 
            branch_labels.append(utils.determine_branch_label(branch, labeltree))

        max_width = np.max(self.medial_axis)

        if not pixel_to_length_ratio: 
            # define the ratio using the assumption that the widest
            # branch is the trunk, and we just assume that it is 
            # about 20cm wide
            trunk_est_width = 200 #mm
            self.pixel_to_length_ratio = max_width / trunk_est_width

        if verbose:
            utils.visualise_result(branches, 'branches', img_shape = self.img.shape,
                                   title = 'Tree branches (before pruning)')

        branch_lengths = {branch_label: cv2.arcLength(branch, False) 
                          for branch, branch_label in zip(branches, branch_labels)}
        max_length = max(branch_lengths.values())
        
        branch_widths = {}
        branch_properties = {}
        for branch, branch_lablel in zip(branches, branch_labels): 
            length = branch_lengths[branch_lablel]
            branch_width = utils.get_branch_width(branch, self.medial_axis)
            branch_widths[branch_lablel] = branch_width

            weighting = utils.decide_branch_weighting(branch_width, max_width, max_length, length, 
                                                      self.pixel_to_length_ratio,
                                                      width_ideal_threshold= (30, 110))

            branch_properties[branch_lablel] = [branch, # branch pixels
                                                length , #branch length
                                                weighting] 

        if verbose: 
            utils.visualise_result(branch_properties, 'branch_weightings', img_shape = self.medial_axis.shape,
                                underlay_img=True, img = self.img)

        edge_list, nodes, loop_edges, inter_nodes = pre_graph(labeltree, branch_properties, intersec_pts, endpts)
        G, max_path = create_graph(edge_list, nodes, inter_nodes)

        if verbose:
            utils.visualise_result(G, 'graph', nodes=nodes,
                                   title = 'Graph representation of bodes before pruning.')

        # lets start by just pruning the very small branches, then everything 
        # else is considered at the analysis stage

        labelisofil_pruned, edge_list_pruned, nodes_pruned, inter_nodes_pruned, G_p, pruned_labels = prune_graph(
            G.copy(), nodes.copy(), edge_list.copy(), labeltree, 
            inter_nodes, loop_edges, max_path, 
            length_thresh= self.pixel_to_length_ratio * 50, weight_thresh = 0.4,
            max_iter=50)
        
        if verbose:
            utils.visualise_result(G_p, 'graph', nodes=nodes ,
                                   title = 'Graph representation of nodes after pruning.')

        
        branch_properties_pruned = branch_properties.copy()
        for branch in list(branch_properties_pruned.keys()): 
            if branch in pruned_labels: 
                branch_properties_pruned.pop(branch)
        
        if verbose: 
            utils.visualise_result(branch_properties_pruned, 'branches', img_shape = self.medial_axis.shape,
                                   title = 'Tree branches (after pruning)')

        self.Graph = G_p
        self.branch_pixels = branch_properties
        self.branch_widths = branch_widths
        self.node_info = nodes

        print("Completed sectioning, graph defining, and pruning.")


    def analyse_branches(self, 
                         pixel_to_length_ratio: float = None, 
                         drone_width: float = None,
                         claw_width: float = None,
                         stride_factor: int = 150,
                         angle_threshold: tuple = (0, 35),
                         width_threshold : tuple = (30, 110),
                         verbose: bool = False):
        
        if pixel_to_length_ratio is None: 
            try: 
                pixel_to_length_ratio = self.pixel_to_length_ratio
            except AttributeError:
                print("No value for pixel_to_length ratio found (passed into funciton or previously calculated).")
                print("CRITICAL error - returning.")
                return 

        # This is where the branch analysis happens

        branches = self.branch_pixels
        branch_widths = self.branch_widths
        G = self.Graph

        for branch in branches: 

            # Analyse the local angle of the branch
            

            # Analyse the local width of the branch 

            # Analyse the branch curvature



            pass

        utils.visualise_curvature(branches, curvatures)


@utils.timeit()
def perf_medial_axis(binary_mask: np.ndarray) -> np.ndarray:
    
    skeleton, distance = medial_axis(binary_mask, return_distance=True)
    distance_on_skeleton = distance * skeleton
    return skeleton, distance_on_skeleton


@utils.timeit()
def prune_graph(G, nodes, edge_list, labelisofil, inter_nodes,
                loop_edges, max_path , 
                length_thresh=0, weight_thresh = 0.2,
                max_iter=1, verbose : bool = False):
    '''
    Function to remove unnecessary branches, while maintaining connectivity
    in the graph. Also updates edge_list, nodes, and inter_nodes.


    Returns
    -------
    labelisofil : list
        Updated from input.
    edge_list : list
        Updated from input.
    nodes : list
        Updated from input.
    inter_nodes : list
        Updated from input.
    G: nx.Graph
        Updated from input.
    removed_labels : list
        List of labels of branches that have been pruned.
    '''

    removed_labels = []

    iterat = 0
    while True:
        degree = dict(G.degree())

        # Remove unconnected nodes
        unconn = [key for key in degree if degree[key] == 0]
        for node in unconn:
            G.remove_node(node)

        single_connect = [key for key in degree if degree[key] == 1]

        delete_candidate = list((set(nodes) - set(max_path[0])) & set(single_connect))

        if not delete_candidate and len(loop_edges[0]) == 0:
            break

        # Gather edges and their corresponding intensities for pruning
        edge_candidates = [(edge[2][0], edge) for edge in edge_list[0]
                            if edge[0] in delete_candidate or edge[1] in delete_candidate]

        # Add loop edges to candidates
        edge_candidates += [(edge[2][0], edge) for edge in loop_edges[0]]

        del_idx = []
        for idx, edge in edge_candidates:
            length = edge[2][1]  # Extract branch length from inter_nodes format
            weight = edge[2][2]
            
            criteria1 = length < length_thresh
            criteria2 = weight < weight_thresh
            # criteria = criteria1 & criteria2
            criteria = criteria1 and criteria2

            if criteria:
                edge_pts = np.where(labelisofil == edge[2][0])
                # x_pts = edge_pts[0]
                # y_pts = edge_pts[1]

                removed_labels.append(edge[2][0])

                labelisofil[edge_pts] = 0  # Remove branch

                try:
                    edge_list[0].remove(edge)
                    nodes.pop(edge[1])
                except ValueError:
                    try: 
                        loop_edges[0].remove(edge)
                    except ValueError:
                        if verbose: 
                            print(f"Node not found in edge list or loop list ({edge})")
                except KeyError:
                    if verbose: 
                        print(f"Edge not found in nodes ({edge[1]})")
                try: 
                    G.remove_edge(edge[0], edge[1])
                except nx.NetworkXError:
                    if verbose: 
                        print(f"Edge not found in graph ({edge})")
                else: 
                    if verbose: 
                        print(f"Removed edge from graph: {edge}")

                del_idx.append(idx)

        # Remove corresponding inter_nodes entries
        # if del_idx:
        #     del_idx.sort()
        #     for idx in del_idx[::-1]:
        #         inter_nodes[0].pop(idx)

        # Merge nodes if necessary
        while True:
            degree = dict(G.degree())
            doub_connect = [key for key in degree if degree[key] == 2]

            if not doub_connect:
                break

            for node in doub_connect:
                G = utils.merge_nodes(node, G) # need to handle metadata !!!!!!!!!!!!!!!!!!!!!!!!

        iterat += 1
        if iterat == max_iter:
            break

    return labelisofil, edge_list, nodes, inter_nodes, G, removed_labels



def create_graph(edge_list, nodes, verbose_nodes)-> nx.Graph:

    G = nx.Graph()
    max_path = []
    extremum = []

    # add nodes
    G.add_nodes_from(nodes.keys())
    
    for edge in edge_list[0]:
        G.add_edge(edge[0], edge[1], weight = edge[2][2], length = edge[2][1], branch_label = edge[2][0])

    paths = dict(nx.shortest_path_length(G))
    values = []
    node_extrema = []

    for i in paths.keys():
        j = max(paths[i].items(), key=operator.itemgetter(1))
        node_extrema.append((j[0], i))
        values.append(j[1])

    max_path_length = max(values)
    start, finish = node_extrema[values.index(max_path_length)]
    extremum.append([start, finish])

    def get_weight(pat):
        return sum([G[x][y]['weight'] for x, y in
                    zip(pat[:-1], pat[1:])])

    all_paths = []
    all_weights = []

    # Catch the weird edges cases where
    max_npath = 500
    for it, pat in enumerate(nx.shortest_simple_paths(G, start, finish)):
        if pat in all_paths:
            break

        if it > 0:
            if all_weights[-1] > get_weight(pat):
                break

        if it > max_npath:
            raise ValueError("Unable to find maximum path. This is likely a bug")
            break

        all_paths.append(pat)
        all_weights.append(get_weight(pat))


    long_path = all_paths[all_weights.index(max(all_weights))]

    max_path.append(long_path)


    return G, max_path



@utils.timeit()
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
    nodes = {}
    edge_list = []
    loop_edges = []



    inter_nodes_temp = []
    # Create end_nodes, which contains lengths, and nodes, which we will
    # later add in the intersections
    end_nodes.append([(labelisofil[labelisofil.shape[0] - end[1] -1, end[0]],
                       branches[labelisofil[labelisofil.shape[0] - end[1] -1, end[0]]][1], # length
                       branches[labelisofil[labelisofil.shape[0] - end[1] -1, end[0]]][2]) # weighting
                      for end in ends])
    # nodes.append([labelisofil[i[0], i[1]] for i in ends])
    nodes = {labelisofil[labelisofil.shape[0] - end[1] -1, end[0]]: end for end in ends}

    # Intersection nodes are given by the intersections points of the filament.
    # They are labeled alphabetically (if len(interpts[n])>26,
    # subsequent labels are AA,AB,...).
    # The branch labels attached to each intersection are included for future
    # use.
    inter_nodes_temp = []
    positions = []

    for intersec in interpts:  # assuming interpts is iterable
        for i in intersec:  
            uniqs = {
                (labelisofil[i[0] + dx, i[1] + dy],  # branch label
                branches[labelisofil[i[0] + dx, i[1] + dy]][1],  # length
                branches[labelisofil[i[0] + dx, i[1] + dy]][2]) # weighting
                # for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                #             ( 0, -1),          ( 0, 1),
                #             ( 1, -1), ( 1, 0), ( 1, 1)]
                for dx, dy in [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                            ( 0, -2), ( 0, -1),          ( 0, 1), ( 0, 2),
                            ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                            ( 2, -2), ( 2, -1), ( 2, 0), ( 2, 1), ( 2, 2)]
                if 0 <= i[0] + dx < labelisofil.shape[1] and 0 <= i[1] + dy < labelisofil.shape[0]
                and labelisofil[i[0] + dx, i[1] + dy] > 0
            }
            if not len(uniqs):
                print("WARNING: Did not find and branches at intersection")
            inter_nodes_temp.append(list(uniqs))
            positions.append((i[1], labelisofil.shape[0] - 1 - i[0]))

    # Add intersection labels
    inter_nodes.append(list(zip(utils.product_gen(string.ascii_uppercase), inter_nodes_temp)))

    for alpha, pos in zip(utils.product_gen(string.ascii_uppercase), positions):
        nodes[alpha] = pos

    # Edges are created from the information contained in the nodes.
    edge_list_temp = []
    loops_temp = []
    for i, inters in enumerate(inter_nodes[0]):
        end_match = list(set(inters[1]) & set(end_nodes[0])) # interpts may also be end points
        for k in end_match:
            edge_list_temp.append((inters[0], k[0], k))

        for j, inters_2 in enumerate(inter_nodes[0]):
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

    return edge_list, nodes, loop_edges, inter_nodes

# def prune_graph()


@utils.timeit()
def find_branchpix(branches, labelfil, final=True, debug=False, remove_region : bool = False):
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
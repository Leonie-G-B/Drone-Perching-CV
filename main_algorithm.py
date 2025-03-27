
from utils import load_in_img, load_in_rdom_imgs
import image_processing as TreeProcessor


####### DEFAULT IMG AND MASK LOCS #########
IMG = "yolo/dataset/images/train/Albizia julibrissin_branch_1 (2).jpg"
MASK = "yolo/dataset/masks/train/Albizia julibrissin_branch_1 (2).png"
###########################################


# Main call point for processing the tree input images

def main(verbose: bool = False, rdom = False):
    # Load in the model

    # Segmentation 

    #### In the meantime while the model is being trained, 
    #### we can just laod in a segmentation example 
    #### and use that to test the rest of the pipeline

    if rdom:
        img_loc, mask_loc = load_in_rdom_imgs("yolo/dataset/images/train", "yolo/dataset/masks/train")
        example_img  = load_in_img(img_loc)
        example_mask = load_in_img(mask_loc)
    else:
        example_img  = load_in_img(IMG)
        example_mask = load_in_img(MASK)
    

    # Initialise the tree object
    # Tree = TreeProcessor.Tree(img= example_img, segmentation= example_mask, resize = 1)
    Tree = TreeProcessor.Tree(img= example_img, segmentation= example_mask, resize = 1.0)

    # Skeletonisation / medial axis 
    Tree.populate_medial_axis(verbose = verbose)

    # Section brances and populate data structure with relevant information
    Tree.section_branches(prune = True, verbose=verbose)

    # Analyse remaining branches/ branch sections for suitability
    Tree.analyse_branches(drone_width = 180, 
                          claw_width= 70,
                          verbose=False)

    # Identify best location and show results
    Tree.rank_branches(angle_lambda= 0.3,
                        curvature_lambda= 0.2,
                        width_lambda = 0.5)
    if verbose: 
        Tree.show_results()

    pass



if __name__ == "__main__":

    main(verbose = False, rdom = False)
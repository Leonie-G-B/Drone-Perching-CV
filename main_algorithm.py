
from utils import load_in_img
import image_processing as TreeProcessor

# Main call point for processing the tree input images

def main():
    # Load in the model

    # Segmentation 

    #### In the meantime while the model is being trained, 
    #### we can just laod in a segmentation example 
    #### and use that to test the rest of the pipeline

    img_loc, mask_loc = load_in_rdom_imgs("yolo/dataset/images/train", "yolo/dataset/masks/train")

    # example_img  = load_in_img(IMG)
    # example_mask = load_in_img(MASK)
    example_img  = load_in_img(img_loc)
    example_mask = load_in_img(mask_loc)


    # Initialise the tree object
    # Tree = TreeProcessor.Tree(img= example_img, segmentation= example_mask, resize = 1)
    Tree = TreeProcessor.Tree(img= example_img, segmentation= example_mask, resize = 1.0)



    # Skeletonisation / medial axis 
    Tree.populate_medial_axis(verbose = False)

    # Section brances and populate data structure with relevant information
    Tree.section_branches()

    # Now can begin eliminating branches that are immediately unsuitable


    # Analyse remaining branches/ branch sections for suitability


    # Identify best location and show results


    pass



if __name__ == "__main__":

    main()
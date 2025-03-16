
from utils import load_in_img
import image_processing as TreeProcessor

# Main call point for processing the tree input images

def main():
    # Load in the model

    # Segmentation 

    #### In the meantime while the model is being trained, 
    #### we can just laod in a segmentation example 
    #### and use that to test the rest of the pipeline
    example_img  = load_in_img("yolo/dataset/images/train/Albizia julibrissin_branch_1 (2).jpg")
    example_mask = load_in_img("yolo/dataset/masks/train/Albizia julibrissin_branch_1 (2).png")

    # Initialise the tree object
    Tree = TreeProcessor.Tree(img= example_img, segmentation= example_mask, resize = 0.5)

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

# Everything to do with training the segmentation model will be controlled from here
# Call all parent functions from here 
# Lines of code can be commented out as required

# IMPORTS 
import pre_training_process as PrePro



def main():
    # Preprocess dataset for training
    PrePro.main(mask_dir='yolo/dataset/masks',
                resize_factor= 0.4,
                dir = 'both')

    # Train model

    # Test model on custom input


    pass



if __name__ == "__main__":
    main()
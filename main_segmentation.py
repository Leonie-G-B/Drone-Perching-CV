
# Everything to do with training the segmentation model will be controlled from here
# Call all parent functions from here 
# Lines of code can be commented out as required

# IMPORTS 
import pre_training_process as PrePro
import model_training as Train


def main():
    # Preprocess dataset for training
    PrePro.main(mask_dir='yolo/dataset/masks',
                resize_factor= 0.2,
                dir = 'both')

    # Train model
    Train.train_yolo(model_dir = "models",
                     model_name= "yolo11n-seg.pt",
                     yaml_name = 'branch_segmentation.yaml',
                     epochs= 2)

    # Test model on custom input



if __name__ == "__main__":
    main()
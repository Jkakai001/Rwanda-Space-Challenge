import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def check_folder_contents(directories):
    """
    Check the contents of each folder and print the number of images found.
    """
    for folder in directories:
        print(f"Checking folder: {folder}")
        for subdir, dirs, files in os.walk(folder):
            print(f"Subdirectory: {subdir}")
            print(f"Number of images: {len(files)}")
            for file in files:
                print(f"Found file: {file}")

def load_multiple_datasets(directories, target_size=(128, 128), batch_size=32):
    """
    Load images from multiple directories and combine them into a single dataset.
    """
    datagen = ImageDataGenerator(rescale=1./255)

    combined_train_images = []
    combined_train_labels = []

    for i, folder_path in enumerate(directories):
        # Check if the directory exists and contains subdirectories
        if not os.path.exists(folder_path):
            print(f"Error: Directory {folder_path} does not exist.")
            continue
        
        # Load data from the directory
        print(f"Loading data from {folder_path}...")
        train_data = datagen.flow_from_directory(
            folder_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',  # Can change to 'categorical' if one-hot encoded labels are needed
            shuffle=False
        )
        
        # Check if images are loaded
        if train_data.samples == 0:
            print(f"No images found in {folder_path}. Please check the structure.")
            continue

        # Collect images and labels from all batches
        for batch_images, batch_labels in train_data:
            combined_train_images.append(batch_images)
            combined_train_labels.append(batch_labels)
            
            # Stop when we've processed all samples in the directory
            if len(combined_train_images) * batch_size >= train_data.samples:
                break
    
    # If images were loaded, concatenate them
    if combined_train_images:
        combined_train_images = np.concatenate(combined_train_images)
        combined_train_labels = np.concatenate(combined_train_labels)
        return combined_train_images, combined_train_labels
    else:
        print("No images were loaded from any dataset.")
        return None, None

# Example usage
folders = [r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\HerbaceousVegetation', 
           r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\River', 
           r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\Residential']

# First, check the folder contents
check_folder_contents(folders)

# Then load the images
train_images, train_labels = load_multiple_datasets(folders)

if train_images is not None:
    print(f"Total images loaded: {train_images.shape[0]}")
else:
    print("No images were loaded.")


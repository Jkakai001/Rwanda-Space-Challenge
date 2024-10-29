from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

def load_multiple_datasets(directories, target_size=(128, 128), batch_size=32):
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
            class_mode='sparse',
            shuffle=False
        )
        
        # Check if images are loaded
        if train_data.samples == 0:
            print(f"No images found in {folder_path}. Please check the structure.")
            continue
        
        # Collect images and labels
        while True:
            try:
                images, labels = next(train_data)
                combined_train_images.append(images)
                combined_train_labels.append(labels)
            except StopIteration:
                break
    
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

train_images, train_labels = load_multiple_datasets(folders)

if train_images is not None:
    print(f"Total images loaded: {train_images.shape[0]}")

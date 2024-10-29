
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data_from_directory(data_dir, target_size=(128, 128), batch_size=32, subset=None):
    """
    Load images from a single directory using ImageDataGenerator.
    Args:
    - data_dir: Directory containing subfolders of images for each class.
    - target_size: Size to resize the images.
    - batch_size: Number of images to return per batch.
    - subset: 'training' or 'validation' (for splitting the dataset).
    
    Returns:
    - data_generator: A data generator for the dataset.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset=subset
    )
    
    return data_generator


def load_multiple_datasets(directories, target_size=(128, 128), batch_size=32):
    """
    Load images from multiple directories, combine them into one dataset.
    Args:
    - directories: List of directories to load data from.
    - target_size: Size to resize images.
    - batch_size: Number of images per batch.
    
    Returns:
    - combined_train_data: Combined training data generator.
    - combined_val_data: Combined validation data generator.
    """
    train_data_list = []
    val_data_list = []
    
    for directory in directories:
        # Load data from each directory
        train_data = load_data_from_directory(directory, target_size, batch_size, subset='training')
        val_data = load_data_from_directory(directory, target_size, batch_size, subset='validation')
        
        train_data_list.append(train_data)
        val_data_list.append(val_data)
    
    # Combine the datasets using numpy
    combined_train_images = np.concatenate([train_data.next()[0] for train_data in train_data_list])
    combined_train_labels = np.concatenate([train_data.next()[1] for train_data in train_data_list])
    
    combined_val_images = np.concatenate([val_data.next()[0] for val_data in val_data_list])
    combined_val_labels = np.concatenate([val_data.next()[1] for val_data in val_data_list])
    
    return (combined_train_images, combined_train_labels), (combined_val_images, combined_val_labels)


# Example usage:
directories = ['E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\HerbaceousVegetation', 'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\River', 'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\Residential']  # Replace with actual paths
(train_images, train_labels), (val_images, val_labels) = load_multiple_datasets(directories)

# Verifying the combined dataset sizes
print("Training data size:", train_images.shape, train_labels.shape)
print("Validation data size:", val_images.shape, val_labels.shape)


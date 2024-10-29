
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check the contents of each folder and print the number of images found
def check_folder_contents(directories):
    for folder in directories:
        print(f"Checking folder: {folder}")
        for subdir, dirs, files in os.walk(folder):
            print(f"Subdirectory: {subdir}")
            print(f"Number of images: {len(files)}")
            for file in files:
                print(f"Found file: {file}")

# Load images from multiple directories and combine them into a single dataset
def load_multiple_datasets(directories, target_size=(128, 128), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    combined_train_images = []
    combined_train_labels = []

    for i, folder_path in enumerate(directories):
        if not os.path.exists(folder_path):
            print(f"Error: Directory {folder_path} does not exist.")
            continue
        
        print(f"Loading data from {folder_path}...")
        train_data = datagen.flow_from_directory(
            folder_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',  # Can change to 'categorical' for one-hot labels
            shuffle=False
        )
        
        if train_data.samples == 0:
            print(f"No images found in {folder_path}. Please check the structure.")
            continue

        for batch_images, batch_labels in train_data:
            combined_train_images.append(batch_images)
            combined_train_labels.append(batch_labels)
            
            if len(combined_train_images) * batch_size >= train_data.samples:
                break
    
    if combined_train_images:
        combined_train_images = np.concatenate(combined_train_images)
        combined_train_labels = np.concatenate(combined_train_labels)
        return combined_train_images, combined_train_labels
    else:
        print("No images were loaded from any dataset.")
        return None, None

# Define the CNN model architecture
def build_model(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Adjust the number of output classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Plot the training and validation accuracy and loss
def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# Paths to the folders containing the images for the dataset
folders = [
    r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\HerbaceousVegetation',
    r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\River',
    r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\Residential'
]

# Check folder contents to ensure images are present
check_folder_contents(folders)

# Load the images and labels from multiple datasets
train_images, train_labels = load_multiple_datasets(folders)

if train_images is not None:
    print(f"Total images loaded: {train_images.shape[0]}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_model(input_shape=(128, 128, 3), num_classes=3)  # Adjust the num_classes as per your dataset
    
    # Train the model with augmented data
    train_datagen = ImageDataGenerator(
        rotation_range=20, 
        zoom_range=0.15, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.15, 
        horizontal_flip=True, 
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Train the model
    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), 
                        epochs=10, 
                        validation_data=test_datagen.flow(X_test, y_test))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot the training and validation results
    plot_training(history)
    
else:
    print("No images were loaded, so training cannot proceed.")


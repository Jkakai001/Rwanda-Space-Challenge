
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_file, target_size=(128, 128), batch_size=32):
    """
    Load images from the directory 
    Return them in batches.
    """
    # Create an instance of ImageDataGenerator for training and validation
    gen_data = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Load the training data
    train_generator = gen_data.flow_from_directory(
        data_file,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',  # Use 'sparse' for sparse categorical labels (integer labels)
        subset='training'
    )
    
    # Loading the validation data
    validation_generator = gen_data.flow_from_directory(
        data_file,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Redirection to the relevant data path
data_file = r"E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT"  # Actual path to the dataset

# Load dataset
train_data, val_data = load_data(data_file)

# Fetch all training data
X_train, y_train = [], []
for batch_images, batch_labels in train_data:
    X_train.append(batch_images)
    y_train.append(batch_labels)
    if len(X_train) * train_data.batch_size >= train_data.samples:
        break

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Fetch all validation data
X_val, y_val = [], []
for batch_images, batch_labels in val_data:
    X_val.append(batch_images)
    y_val.append(batch_labels)
    if len(X_val) * val_data.batch_size >= val_data.samples:
        break

X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

# Image data augmentation
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

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(13, activation='softmax')  # Adjust the number of output classes
])

# Model Compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=20, 
                    validation_data=test_datagen.flow(X_val, y_val))

# Model Evaluation
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_acc:.4f}")

# Making predictions on validation set
predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Outputting the predictions and confidence scores
for i, (pred, conf) in enumerate(zip(predicted_classes, confidence_scores)):
    print(f"Image {i+1}: Class = {pred}, Confidence = {conf:.4f}")

# Plotting the training & validation accuracy and loss
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

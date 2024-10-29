import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
# def load_data():
#     # Placeholder: Replace with actual data loading
#     # Example: Load images and labels (vegetation, water, urban house)
#     images = np.random.random((100, 128, 128, 3))  # 100 random images
#     labels = np.random.randint(3, size=(100,))  # Random labels for 3 classes
#     return images, labels
def load_data(data_file, target_size=(128, 128), batch_size=32):
    """
    Load images from the directory 
    Return them in batches.
    .
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
#redirection to the relevant data path
data_file = "E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT" # actual path to the dataset
train_data, val_data = load_data(data_file)

images, labels = load_data()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Image data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
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
    Dense(3, activation='softmax')  # 3 output classes (vegetation, water, urban house)
])

# Model Compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=test_datagen.flow(X_test, y_test))

# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Making predictions on test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Outputing the  predictions and confidence scores
for i, (pred, conf) in enumerate(zip(predicted_classes, confidence_scores)):
    print(f"Image {i+1}: Class = {pred}, Confidence = {conf:.4f}")

# Ploting the  training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()
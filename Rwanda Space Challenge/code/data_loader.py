
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    
    # Load the validation data
    validation_generator = gen_data.flow_from_directory(
        data_file,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Example usage
data_file = '/path/to/your/dataset'  # Replace with the actual path to your dataset
train_data, val_data = load_data(data_file)

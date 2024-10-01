import os
import numpy as np
import matplotlib.image as img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import List, Tuple

def load_images(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from the specified train and test directories.

    Parameters:
    train_path (str): Path to the training data directory.
    test_path (str): Path to the testing data directory.

    Returns:
    tuple: A tuple containing two NumPy arrays: train_files and test_files.
    """
    train_filenames = os.listdir(train_path)
    train_files: List = []
    for filename in train_filenames:
        sample = img.imread(os.path.join(train_path, filename), 'r')
        train_files.append(sample)
    
    test_filenames = os.listdir(test_path)
    test_files: List = []
    for filename in test_filenames:
        sample = img.imread(os.path.join(test_path, filename), 'r')
        test_files.append(sample)
    
    return np.array(train_files), np.array(test_files)

def setup_image_data_generator(train: np.ndarray) -> ImageDataGenerator:
    """
    Initialize and fit an ImageDataGenerator on the training data.

    Parameters:
    train (np.ndarray): The training data.

    Returns:
    ImageDataGenerator: The initialized and fitted data generator.
    """
    datagen = ImageDataGenerator()
    datagen.fit(train)
    return datagen

if __name__ == "__main__":
    train_path = 'D:/MRIdata/train/'
    test_path = 'D:/MRIdata/test/'
    
    # Ensure the directories exist before loading
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: One of the provided paths does not exist:\nTrain path: {train_path}\nTest path: {test_path}")
    else:
        # Load images
        train_files, test_files = load_images(train_path, test_path)
        
        # Reshape for model input
        train_files = train

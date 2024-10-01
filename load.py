
import os
import matplotlib.image as img
from typing import List, Tuple

def load_images(train_path: str, test_path: str) -> Tuple[List, List]:
    """
    Load images from the specified train and test directories.

    Parameters:
    train_path (str): Path to the training data directory.
    test_path (str): Path to the testing data directory.

    Returns:
    tuple: A tuple containing two lists: train_files and test_files.
           train_files (list): List of images from the training directory.
           test_files (list): List of images from the testing directory.
    """
    
    # Load training images
    train_filenames = os.listdir(train_path)
    train_files: List = []
    for filename in train_filenames:
        sample = img.imread(os.path.join(train_path, filename), 'r')
        train_files.append(sample)
    
    # Load testing images
    test_filenames = os.listdir(test_path)
    test_files: List = []
    for filename in test_filenames:
        sample = img.imread(os.path.join(test_path, filename), 'r')
        test_files.append(sample)
    
    return train_files, test_files


if __name__ == "__main__":
    train_path = 'D:/MRIdata/train/'
    test_path = 'D:/MRIdata/test/'
    
    # Ensure the directories exist before loading
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: One of the provided paths does not exist:\nTrain path: {train_path}\nTest path: {test_path}")
    else:
        train_files, test_files = load_images(train_path, test_path)
        print(f"Loaded {len(train_files)} training images and {len(test_files)} testing images.")
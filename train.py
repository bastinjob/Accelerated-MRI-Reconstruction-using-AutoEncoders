import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_autoencoder
from dataset import load_dataset
from preprocess import preprocess_data

def train_autoencoder(train_data: np.ndarray, test_data: np.ndarray, epochs: int = 50, batch_size: int = 32):
    """
    Train the autoencoder and visualize the training process.

    Parameters:
    train_data (np.ndarray): Training data.
    test_data (np.ndarray): Testing data.
    epochs (int): Number of epochs to train the model.
    batch_size (int): Batch size during training.

    Returns:
    history: Training history object.
    """

    # Create the autoencoder
    input_shape = train_data.shape[1:]  # Assuming (192, 192, 1) shape for images
    autoencoder = create_autoencoder(input_shape)

    # Image Data Generator for data augmentation
    datagen = ImageDataGenerator()
    datagen.fit(train_data)

    # Train the autoencoder
    history = autoencoder.fit(
        datagen.flow(train_data, train_data, batch_size=batch_size),
        validation_data=(test_data, test_data),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return history

def visualize_training(history):
    """
    Visualize the training loss and metrics.

    Parameters:
    history: The history object returned by the fit method.
    """
    # Plot training & validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation MSE (Mean Squared Error)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the dataset
    train_path = 'D:/MRIdata/train/'
    test_path = 'D:/MRIdata/test/'
    
    train_data, test_data = load_dataset(train_path, test_path)
    train_data, test_data = preprocess_data(train_data, test_data)

    # Train the autoencoder
    history = train_autoencoder(train_data, test_data, epochs=50, batch_size=32)

    # Visualize the training results
    visualize_training(history)

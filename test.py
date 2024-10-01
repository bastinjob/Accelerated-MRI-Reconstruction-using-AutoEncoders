import numpy as np
import matplotlib.pyplot as plt
from model import create_autoencoder
from dataset import load_dataset
from preprocess import preprocess_data

def evaluate_autoencoder(model, test_data: np.ndarray):
    """
    Evaluate the autoencoder on test data and visualize reconstruction.

    Parameters:
    model: Trained autoencoder model.
    test_data (np.ndarray): Test data to evaluate the model.

    Returns:
    loss, mse: Evaluation loss and mean squared error on test data.
    """
    # Evaluate the model
    loss, mse = model.evaluate(test_data, test_data, verbose=1)
    print(f"Test Loss: {loss}")
    print(f"Test MSE: {mse}")
    
    return loss, mse

def visualize_reconstruction(model, test_data: np.ndarray, n_images: int = 5):
    """
    Visualize original and reconstructed images.

    Parameters:
    model: Trained autoencoder model.
    test_data (np.ndarray): Test data containing images.
    n_images (int): Number of images to visualize.
    """
    # Select a few images from the test data
    idxs = np.random.randint(0, len(test_data), n_images)
    test_samples = test_data[idxs]

    # Get the reconstructed images
    reconstructed_images = model.predict(test_samples)

    # Plot original and reconstructed images
    plt.figure(figsize=(12, 4))
    for i in range(n_images):
        # Display original image
        plt.subplot(2, n_images, i + 1)
        plt.imshow(test_samples[i].reshape(192, 192), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Display reconstructed image
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed_images[i].reshape(192, 192), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the dataset
    train_path = 'D:/MRIdata/train/'
    test_path = 'D:/MRIdata/test/'

    _, test_data = load_dataset(train_path, test_path)
    _, test_data = preprocess_data(_, test_data)

    # Create and load the autoencoder model
    input_shape = (192, 192, 1)  # Assuming input shape
    autoencoder = create_autoencoder(input_shape)

    # Load the pre-trained model weights
    autoencoder.load_weights('autoencoder_weights.h5')  # Assuming you saved the model weights

    # Evaluate the autoencoder on test data
    evaluate_autoencoder(autoencoder, test_data)

    # Visualize some original vs reconstructed images
    visualize_reconstruction(autoencoder, test_data, n_images=5)

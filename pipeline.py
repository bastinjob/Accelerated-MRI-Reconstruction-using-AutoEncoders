import os
from dataset import load_dataset
from preprocess import preprocess_data
from model import create_autoencoder
from train import train_autoencoder, plot_training_history
from test import evaluate_autoencoder, visualize_reconstruction

def run_pipeline(dataset_path: str):
    """
    Orchestrates the entire end-to-end process from data loading, preprocessing, 
    model training, evaluation, and visualization.

    Parameters:
    dataset_path (str): Path to the dataset folder which contains 'train' and 'test' directories.
    """
    # Ensure the dataset path exists
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError("Dataset path is invalid. Ensure 'train' and 'test' directories exist inside the dataset path.")

    # Step 1: Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    train_data, test_data = load_dataset(train_path, test_path)
    train_data, test_data = preprocess_data(train_data, test_data)
    print("Dataset loaded and preprocessed.")

    # Step 2: Create the autoencoder model
    input_shape = (192, 192, 1)
    autoencoder = create_autoencoder(input_shape)

    # Step 3: Train the model
    print("Training the autoencoder model...")
    history = train_autoencoder(autoencoder, train_data, test_data, batch_size=32, epochs=50)
    print("Model training complete.")

    # Plot training history
    plot_training_history(history)

    # Save the trained model weights
    autoencoder.save_weights('autoencoder_weights.h5')
    print("Model weights saved to 'autoencoder_weights.h5'.")

    # Step 4: Evaluate the model
    print("Evaluating the autoencoder on test data...")
    evaluate_autoencoder(autoencoder, test_data)

    # Step 5: Visualize reconstructed images
    print("Visualizing reconstruction...")
    visualize_reconstruction(autoencoder, test_data)

if __name__ == "__main__":
    # Input: Path to the dataset
    dataset_path = 'D:/MRIdata/'  # Example dataset path

    # Run the pipeline
    run_pipeline(dataset_path)

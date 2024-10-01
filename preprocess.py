import numpy as np

def rand_bin_array(K: float, N: int) -> np.ndarray:
    """
    Generate a random binary array where a portion (K) is set to 1 and the rest is 0.

    Parameters:
    K (float): Proportion of the array to be filled with 1's.
    N (int): Dimension of the array (NxN).

    Returns:
    np.ndarray: A binary array of size NxN.
    """
    arr = np.zeros(N*N)
    k = int(K * N * N)
    arr[:k] = 1
    np.random.shuffle(arr)
    return arr.reshape((N, N))

def preprocess_images(train: np.ndarray, K: float = 0.25) -> np.ndarray:
    """
    Apply binary mask transformation on training images.

    Parameters:
    train (np.ndarray): The training images array.
    K (float): Proportion for the binary masking process.

    Returns:
    np.ndarray: Transformed training images.
    """
    train_y = train.copy()  # Optionally save a copy
    for i in range(len(train)):
        train[i] = np.dot(train[i].reshape(192, 192), rand_bin_array(K, 192)).reshape(192, 192, 1)
    return train

if __name__ == "__main__":
    # Example usage of preprocess_images
    dummy_train = np.random.rand(30, 192, 192, 1)  # Replace with actual train data
    preprocessed_train = preprocess_images(dummy_train, K=0.25)
    
    print(f"Preprocessing complete. Preprocessed data shape: {preprocessed_train.shape}")

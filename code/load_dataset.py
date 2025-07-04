# ---------------------------------------------------------------------------------------------
# This script loads the dataset (cifar10, mnist, or medmnist) and prepares it for training.
# ---------------------------------------------------------------------------------------------

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import medmnist
from medmnist.dataset import ChestMNIST
from medmnist.info import INFO

# Note: Make sure you have medmnist and scikit-learn installed.
# pip install medmnist scikit-learn

# Create the custom function to load datasets
def load_dataset(dataset_name: str, val_split_size: float = 0.2, random_state: int = 42, caption: bool = True) -> tuple:
    """
    Loads and preprocesses a specific dataset.

    This function handles loading CIFAR-10 and MNIST from Keras,
    and MedMNIST datasets from the medmnist library.
    For CIFAR-10 and MNIST, it automatically creates a validation set
    from the training data.

    Args:
        dataset_name (str): The name of the dataset to load.
                            Valid options: 'cifar10', 'mnist', 'chestmnist'.
        val_split_size (float): The proportion of the training set to reserve for validation.

    Returns:
        tuple:  A tuple containing the training, validation, and test data.
                Format: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    print(f"Loading dataset: {dataset_name}")

    if dataset_name == 'cifar10':
        # Load CIFAR-10 directly from Keras
        (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10 # Cifar 10 has 10 classes
        # Create a validation set from the training data
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=val_split_size,
            random_state=random_state, # for reproducibility
            stratify=y_train_full # ensure same class distribution
        )

        # Transform labels to categorical format
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

    elif dataset_name == 'mnist':
        # Load MNIST directly from Keras
        (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10 # MNIST has 10 classes (digits 0-9)
        # Create a validation set from the training data
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=val_split_size,
            random_state=random_state, # for reproducibility
            stratify=y_train_full # ensure same class distribution
        )
        # Add the channel dimension (grayscale)
        x_train, x_val, x_test = [np.expand_dims(d, -1) for d in (x_train, x_val, x_test)]

        # Reconfigure the shape of the labels
        y_train, y_val, y_test = [y.reshape(-1, 1) for y in (y_train, y_val, y_test)]

        # Transform labels to categorical format
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

    elif dataset_name == 'chestmnist':
        # Info to know the number of classes
        info = INFO
        num_classes = len(info['chestmnist']['label'])

        # Load Chest-MedMNIST dataset: Already split into train, val, and test sets
        data_train = ChestMNIST(split='train', download=True)
        data_val = ChestMNIST(split='val', download=True)
        data_test = ChestMNIST(split='test', download=True)

        # Extract images and labels
        x_train, y_train = data_train.imgs, data_train.labels
        x_val, y_val = data_val.imgs, data_val.labels
        x_test, y_test = data_test.imgs, data_test.labels

        # Add the channel dimension (grayscale)
        x_train, x_val, x_test = [np.expand_dims(d, -1) for d in (x_train, x_val, x_test)]

    else:
        raise ValueError(f"Dataset name '{dataset_name}' is not valid."
                        "Use 'cifar10', 'mnist', or 'chestmnist'.")

    # Common Preprocessing
    # Normalize images to the [0, 1] range and convert to float32
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Configure target vectors to float32
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    y_test = y_test.astype('float32')

    # Show the shapes of the loaded data if caption is True
    if caption:
        print("Shapes of loaded data:")
        print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val:   {x_val.shape}, y_val:   {y_val.shape}")
        print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")
        print("-" * 30 + "\n")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == '__main__':
    # Example Usage
    print("Demonstration of the dataset loading function.\n")

    # Load MNIST
    (x_train_mnist, y_train_mnist), (x_val_mnist, y_val_mnist), (x_test_mnist, y_test_mnist) = load_dataset('mnist')

    # Load CIFAR-10
    (x_train_cifar, y_train_cifar), (x_val_cifar, y_val_cifar), (x_test_cifar, y_test_cifar) = load_dataset('cifar10')

    # Load Chest-MedMNIST
    (x_train_med, y_train_med), (x_val_med, y_val_med), (x_test_med, y_test_med) = load_dataset('chestmnist')

    print("All datasets loaded successfully!")
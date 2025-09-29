import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer



def data_preparing(target_size: int = 64):
    """
    Load and preprocess CIFAR-10 dataset:
    - Resize images from 32x32 to target_size x target_size
    - Normalize pixel values to [0, 1]
    - One-hot encode labels

    Args:
        target_size (int): Target image dimension (default: 64)

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    # Load original CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Resize using vectorized approach (much faster)
    x_train_resized = np.array([cv2.resize(img, (target_size, target_size)) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, (target_size, target_size)) for img in x_test])

    # One-hot encode labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    return x_train_resized, y_train, x_test_resized, y_test
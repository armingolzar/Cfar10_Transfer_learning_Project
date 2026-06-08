import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Input, Flatten, Dense, Dropout, BatchNormalization,
    RandomFlip, RandomRotation, RandomTranslation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_network(input_shape=(64, 64, 3), num_classes=10):
    """
    Build a transfer learning model for CIFAR-10 using VGG16 (frozen up to block3_pool).
    Includes on-GPU data augmentation and strong regularization.
    
    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
    
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # On-GPU Data Augmentation (applied only during training)
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)          # ~10 degrees
    x = RandomTranslation(0.1, 0.1)(x)  # 10% shift in height/width
    
    # Load pretrained VGG16 (frozen)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # Extract features up to block3_pool (as in your original design)
    base_output = base_model.get_layer('block3_pool').output
    feature_extractor = Model(inputs=base_model.input, outputs=base_output)
    
    # Apply feature extractor to augmented input
    features = feature_extractor(x, training=False)
    
    # Custom classification head (your architecture)
    x = Flatten()(features)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
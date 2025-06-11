# --------------------------------------------------------------------------------------
# This script defines various CNN block implementations using TensorFlow and Keras.
# --------------------------------------------------------------------------------------

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Define block implementations
def vgg_block(input_tensor, filters, kernel_size, layer_name):
    """
    VGG-style block with two convolutional layers followed by max pooling.
    Args:
        input_tensor: Input tensor to the block.
        filters: Number of filters for the convolutional layers.
        kernel_size: Size of the convolutional kernels.
        layer_name: Name prefix for the layers in this block.
    Returns:
        Output tensor after applying the VGG block.
    """
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', name=f"{layer_name}_1")(input_tensor)
    x = layers.BatchNormalization(name=f"{layer_name}_bn_1")(x)  # Add Batch Normalization
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', name=f"{layer_name}_2")(x)
    x = layers.BatchNormalization(name=f"{layer_name}_bn_2")(x)  # Add Batch Normalization
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{layer_name}_3")(x)
    return x

def resnet_block(input_tensor, filters, kernel_size, layer_name):
    """
    ResNet-style block with two convolutional layers and a skip connection.
    Args:
        input_tensor: Input tensor to the block.
        filters: Number of filters for the convolutional layers.
        kernel_size: Size of the convolutional kernels.
        layer_name: Name prefix for the layers in this block.
    Returns:
        Output tensor after applying the ResNet block.
    """
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', name=f"{layer_name}_1")(input_tensor)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', name=f"{layer_name}_2")(input_tensor)
    x = layers.BatchNormalization(name=f"{layer_name}_bn_1")(x)  # Add Batch Normalization
    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"{layer_name}_3")(x)
    x = layers.BatchNormalization(name=f"{layer_name}_bn_2")(x)  # Add Batch Normalization
    x = layers.Add(name=f"{layer_name}_4")([x, shortcut])
    x = layers.Activation('relu', name=f"{layer_name}_5")(x)
    return x

def inception_block(input_tensor, kernel_size, layer_name):
    """
    Inception-style block with multiple convolutional branches.
    Args:
        input_tensor: Input tensor to the block.
        kernel_size: Size of the convolutional kernels for the branches.
        layer_name: Name prefix for the layers in this block.
    Returns:
        Output tensor after applying the Inception block.
    """
    branch1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu', name=f"{layer_name}_1")(input_tensor)
    branch2 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu', name=f"{layer_name}_2")(input_tensor)
    branch2 = layers.Conv2D(64, kernel_size, padding='same', activation='relu', name=f"{layer_name}_3")(branch2)
    branch3 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu', name=f"{layer_name}_4")(input_tensor)
    branch3 = layers.Conv2D(64, kernel_size=(kernel_size[0] + 2, kernel_size[1] + 2), padding='same', activation='relu', name=f"{layer_name}_5")(branch3)
    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=f"{layer_name}_6")(input_tensor)
    branch4 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu', name=f"{layer_name}_7")(branch4)
    return layers.Concatenate(name=f"{layer_name}_8")([branch1, branch2, branch3, branch4])

def densenet_block(input_tensor, filters, kernel_size, layer_name):
    """
    DenseNet-style block with a convolutional layer followed by concatenation.
    Args:
        input_tensor: Input tensor to the block.
        filters: Number of filters for the convolutional layer.
        kernel_size: Size of the convolutional kernel.
        layer_name: Name prefix for the layers in this block.
    Returns:
        Output tensor after applying the DenseNet block.
    """
    x = layers.Conv2D(filters, kernel_size, padding='same', name=f"{layer_name}_1")(input_tensor)
    x = layers.BatchNormalization(name=f"{layer_name}_2")(x)
    x = layers.ReLU(name=f"{layer_name}_3")(x)
    return layers.Concatenate(name=f"{layer_name}_4")([input_tensor, x])


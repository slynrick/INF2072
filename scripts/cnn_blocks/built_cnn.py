# -------------------------------------------------------------------------------------------------------
# This script defines a CNN model using various block types such as VGG, ResNet, Inception, and DenseNet.
# -------------------------------------------------------------------------------------------------------

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Import custom block implementations
from scripts.cnn_blocks.cnn_blocks import vgg_block, resnet_block, inception_block, densenet_block

# Define CNN model
def build_cnn(layers_config, input_shape, num_classes):
    input_tensor = layers.Input(shape=input_shape[1:], name="input")
    x = input_tensor
    for idx, layer in enumerate(layers_config):
        if layer == 'vgg_block_3x3':
            x = vgg_block(x, 64, (3, 3), layer_name=f"vgg_block_3x3_{idx}")
        elif layer == 'vgg_block_5x5':
            x = vgg_block(x, 64, (5, 5), layer_name=f"vgg_block_5x5_{idx}")
        elif layer == 'resnet_block_3x3':
            x = resnet_block(x, 64, (3, 3), layer_name=f"resnet_block_3x3_{idx}")
        elif layer == 'resnet_block_5x5':
            x = resnet_block(x, 64, (5, 5), layer_name=f"resnet_block_5x5_{idx}")
        elif layer == 'inception_block_3x3':
            x = inception_block(x, (3, 3), layer_name=f"inception_block_3x3_{idx}")
        elif layer == 'inception_block_5x5':
            x = inception_block(x, (5, 5), layer_name=f"inception_block_5x5_{idx}")
        elif layer == 'densenet_block_3x3':
            x = densenet_block(x, 32, (3, 3), layer_name=f"densenet_block_3x3_{idx}")
        elif layer == 'densenet_block_5x5':
            x = densenet_block(x, 32, (5, 5), layer_name=f"densenet_block_5x5_{idx}")
        elif layer == 'no_op':
            x = layers.Lambda(lambda y: y, name=f"ide_{idx}")(x)
    x = layers.Flatten(name="flatten")(x)
    output_tensor = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model = models.Model(inputs=[input_tensor], outputs=[output_tensor], name='cnn_model')
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
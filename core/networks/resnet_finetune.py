from typing import Tuple

import tensorflow as tf
import pdb

def resnetfinetune(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    num_classes = output_shape[0]
    preprocess = tf.keras.applications.resnet_v2.preprocess_input
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    base_model.trainable = True

    inputs = tf.keras.layers.Input(input_shape)
    pool    = tf.keras.layers.GlobalAveragePooling2D()
    flatten = tf.keras.layers.Flatten()
    softmax   = tf.keras.layers.Dense(num_classes, activation='softmax')

    x = inputs
    x = preprocess(x)
    x = base_model(x)
    x = pool(x)
    x = flatten(x)
    out = softmax(x)

    #fine tuning (last resnet block)
    for layer in base_model.layers:
        if layer.name.startswith('conv5_block3') and not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    
    return tf.keras.Model(inputs=inputs, outputs=out)
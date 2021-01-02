from typing import Tuple

import tensorflow as tf
import pdb

def resnetfinetune(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    num_classes = output_shape[0]
    model = tf.keras.models.Sequential()
    conv_base = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    conv_base.trainable = True
    model.add(conv_base)
    model.add(tf.keras.layers.Dense(
        num_classes,
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer = 'l2',
        bias_regularizer = 'l2'
    ))
    model.add(tf.keras.layers.Activation('softmax', dtype='float32'))

    #fine tuning (last resnet block)
    for layer in conv_base.layers:
        if layer.name.startswith('conv5_block3') and not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    return model
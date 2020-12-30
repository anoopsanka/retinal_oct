from typing import Tuple

import tensorflow as tf

def resnet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    num_classes = output_shape[0]
    model = tf.keras.models.Sequential()
    conv_base = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    model.add(conv_base)
    model.add(tf.keras.layers.Dense(
        num_classes,
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer = 'l2',
        bias_regularizer = 'l2'
    ))
    model.add(tf.keras.layers.Activation('softmax', dtype='float32'))
    return model
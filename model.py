import tensorflow as tf


def convnet_model(input_shape=(224, 244), include_top=False, pooling='avg'):
    return tf.keras.applications.ResNet50V2(weights='imagenet', include_top=include_top, input_shape=input_shape, pooling=pooling)

def resnet_model(num_classes, input_shape=(224,224,3)):
    model = tf.keras.models.Sequential()
    conv_base = convnet_model(input_shape, include_top=False, pooling='avg')
    model.add(conv_base)
    conv_base.trainable = False
    #model.add(tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax'))
    model.add(tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer='l2',
        bias_regularizer='l2',
        name='last_dense'))
    model.add(tf.keras.layers.Activation('softmax', dtype='float32'))

    return model
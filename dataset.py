import tensorflow as tf
import os

def get_data_generators(data_dir, batch_size, image_size):

    train_datagen_kwargs = dict(rescale=1./255)
    dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size, class_mode='categorical')

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**train_datagen_kwargs)
    train_data = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'), shuffle=True, **dataflow_kwargs)

    test_datagen_kwargs = dict(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_datagen_kwargs)
    validation_data = test_datagen.flow_from_directory(os.path.join(data_dir,'val'), shuffle=False, **dataflow_kwargs)
    test_data = test_datagen.flow_from_directory(os.path.join(data_dir, 'test'), shuffle=False, **dataflow_kwargs)

    return (train_data, validation_data, test_data)


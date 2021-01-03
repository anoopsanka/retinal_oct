from typing import Tuple
import tensorflow as tf
from core.datasets.dataset import Dataset
import tensorflow_datasets as tfds
import pdb
from core.datasets.data_util import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

_TFDS_DATASET_NAME='RetinaDataset'
_NUM_CLASSES=4
CLASS_WEIGHTS = {0: 0.7935887487875849, 1: 2.4252727057149635, 2: 1.8407802375809936, 3: 0.5604348183462108}
class RetinaDatasetWrapper(Dataset):
    def __init__(self, target_image_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.train = self.validation = self.test = None

        self.input_shape = target_image_shape
        self.num_classes = (_NUM_CLASSES,)

    def load(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.train, train_info = tfds.load(_TFDS_DATASET_NAME, split='train[:98%]', shuffle_files=True, as_supervised=True, with_info=True)
        self.validation, validation_info = tfds.load(_TFDS_DATASET_NAME, shuffle_files=True, split='train[-2%:]', as_supervised=True, with_info=True)
        self.test, test_info = tfds.load(_TFDS_DATASET_NAME, split='test', as_supervised=True, with_info=True)

        return self.train, self.validation, self.test

    def get_class_weights(self):
        return CLASS_WEIGHTS  # avoiding computation, should be saved in ds meta info per version

    def prepare(self):
        self.train = self.train.map(self.augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.validation = self.validation.map(self.resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test = self.test.map(self.resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #data_augmentation = tf.keras.Sequential([
         #       tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
          #      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
           #     tf.keras.layers.experimental.preprocessing.RandomZoom(0.3, 0.3)
            #    ])

        #self.train = self.train.map(self.augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)


        return self.train, self.validation, self.test

    def resize_image(self, image, label):
        return tf.image.resize(image, (self.input_shape[0],self.input_shape[1])), label

    def augment_image(self, image, label):
        image, label = self.resize_image(image, label)
        return preprocess_for_train(image, height=self.input_shape[0], width=self.input_shape[1]), label

    def _pre_process(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32) / 255.
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]])

        
        return image, label


    def __repr__(self):
        return (
            'Retina_Dataset_basic_pre_process\n'
            f'Num classes: {self.num_classes}\n'
            f'Input Shape: {self.input_shape}\n'
        )


def main():
    dataset = RetinaDatasetWrapper()
    train, val, test = dataset.load()

    assert train is None
    assert val is None
    assert test is None


if __name__ == '__main__':
    main()
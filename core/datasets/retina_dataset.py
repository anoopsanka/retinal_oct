from typing import Tuple
import tensorflow as tf
from core.datasets.dataset import Dataset

TRAIN_DATA_DIRNAME = Dataset.data_dirname() / "train"
VAL_DATA_DIRNAME = Dataset.data_dirname() / "val"
TEST_DATA_DIRNAME = Dataset.data_dirname() / "test"

class RetinaDataset(Dataset):
    def __init__(self, batch_size:int = 32, target_size: Tuple[int, int] = (224, 224)):
        self.train = self.validation = self.test = None

        self.input_shape = (224, 224, 3)
        self.num_classes = 4
        self.output_shape = (self.num_classes,)

        self.train_datagen_kwargs = dict(rescale=1./255)
        self.test_datagen_kwargs = dict(rescale=1./255)

        self.dataflow_kwargs = dict(target_size=target_size, batch_size=batch_size, seed=42, class_mode='categorical')

    def load_or_generate_data(self):
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self.train_datagen_kwargs)
            self.train = train_datagen.flow_from_directory(TRAIN_DATA_DIRNAME, shuffle=True, **self.dataflow_kwargs)            

    
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**self.test_datagen_kwargs)
            self.validation = test_datagen.flow_from_directory(VAL_DATA_DIRNAME, shuffle=False, **self.dataflow_kwargs)
            self.test = test_datagen.flow_from_directory(TEST_DATA_DIRNAME, shuffle=False, **self.dataflow_kwargs)

    def __repr__(self):
        return (
            'Retina Dataset\n'
            f'Num classes: {self.num_classes}\n'
            f'Input Shape: {self.input_shape}\n'
        )


def main():
    dataset = RetinaDataset(batch_size=32)
    dataset.load_or_generate_data()


if __name__ == '__main__':
    main()
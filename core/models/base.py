from pathlib import Path
from typing import Callable, Dict

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model as KerasModel
import tensorflow_datasets as tfds
from core.models.model_util import get_optimizer, get_loss

WEIGHTS_DIRNAME = Path(__file__).parents[1].resolve() / "weights"
_SEED=42
DEFAULT_CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
class Model:

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = {}
    ):
        tf.random.set_seed(_SEED)
        np.random.seed(_SEED)

        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        self.data = dataset_cls(**dataset_args)

        self.network = network_fn(input_shape=self.data.input_shape, output_shape=self.data.num_classes)

    @property
    def weights_filename(self) -> str:
        WEIGHTS_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIRNAME / f"{self.name}_weights.h5")

    @property
    def image_shape(self):
        return self.data.input_shape

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, lr: float = 1e-3, optimizer: str = 'crossentropy', loss: str = 'adam', verbose: int = 1, callbacks: list = []):
        dataset.train, dataset.validation , dataset.test = dataset.prepare()
        self.network.compile(loss=get_loss(loss), optimizer=get_optimizer(optimizer, lr=lr), metrics=self.metrics())

        if optimizer == 'crossentropy':
            class_weight = dataset.get_class_weights()
        else:
            class_weight =DEFAULT_CLASS_WEIGHTS

        fit_kwargs = dict(
                        epochs = epochs,
                        validation_data = dataset.validation.batch(batch_size),
                        verbose = verbose,
                        callbacks = callbacks,
                        class_weight = class_weight
                    )
        self.network.fit(dataset.train.batch(batch_size), **fit_kwargs)

    def evaluate(self, data, batch_size: int = 32, verbose=1):
        return self.network.evaluate(data.batch(batch_size), verbose=verbose)

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
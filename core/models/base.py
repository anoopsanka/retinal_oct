from pathlib import Path
from typing import Callable, Dict

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import RMSprop

WEIGHTS_DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model:

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = {}
    ):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        self.data = dataset_cls(**dataset_args)

        self.network = network_fn(self.data.input_shape, self.data.output_shape)

    @property
    def weights_filename(self) -> str:
        WEIGHTS_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIRNAME / f"{self.name}_weights.h5")

    @property
    def image_shape(self):
        return self.data.input_shape

    def fit(self, dataset, epochs: int = 10, verbose: int = 1, callbacks: list = []):

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        fit_kwargs = dict(
                        steps_per_epoch = dataset.train.samples // dataset.train.batch_size,
                        validation_steps = dataset.validation.samples // dataset.validation.batch_size,
                        epochs = epochs,
                        validation_data = dataset.validation,
                        verbose = verbose,
                        callbacks = callbacks
                    )
        self.network.fit(dataset.train, **fit_kwargs)

    def evaluate(self, data, verbose=2):
        self.network.evaluate(data, verbose=verbose)



    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self):
        return RMSprop()

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
'''Utility to train the model'''
from time import time

from tensorflow.keras.callbacks import EarlyStopping, Callback

from core.datasets.dataset import Dataset
from core.models.base import Model
import wandb
from wandb.keras import WandbCallback

EARLY_STOPPING = True

def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, lr: float, loss: str, optimizer:str, use_wandb: bool = False) -> Model:
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if use_wandb:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    model.network.summary()

    t= time()
    _history = model.fit(dataset=dataset, epochs=epochs, batch_size=batch_size, lr=lr, loss=loss, optimizer=optimizer, callbacks=callbacks)
    print ('Training took {:2f} s'.format(time() - t))

    return model
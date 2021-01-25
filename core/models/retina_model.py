
from typing import Dict, Callable

from core.datasets.retina_dataset import RetinaDataset
from core.models.base import Model
from core.networks import resnetconv

class RetinaModel(Model):


    def __init__(
        self,
        dataset_cls: type = RetinaDataset,
        network_fn: Callable = resnetconv,
        dataset_args: Dict = {}
    ):
        print (dataset_cls)
        super().__init__(dataset_cls, network_fn, dataset_args)
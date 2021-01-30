import argparse
import importlib
import json
import os
from typing import Dict
from training.gpu_manager import GPUManager
import wandb
import tensorflow as tf
import numpy as np
import random
from core.models.simclr_model import Pretrained_SimCLR_Model
from core.models.model_utils.lr_schedule import WarmUpAndCosineDecay
from core.datasets.data_augmentation import train_classification_aug
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import wandb
from wandb.keras import WandbCallback


DEFAULT_TRAIN_ARGS = {"learning_rate": 0.01,
     "epochs": 1,
     "learning_rate_scaling": "linear",
     "warmup_epochs": 10,
     "batch_size": 256,
     "num_classes": 4,
     "use_blur": True,
     "proj_head_mode": "nonlinear",
     "proj_out_dim" : 128,
     "num_proj_layers": 3,
     "ft_proj_selector": 0,
     "resnet_depth": 18,
     "resnet_width_multiplier": 1,
     "resnet_se_ratio": 0.0,
     "resnet_sk_ratio": 0.0,
     "hidden_norm": True,
     "temperature" :1.0,
     "IMG_SIZE": 128}

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)



def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = True):
    """
    Parameters
    -------------
    experiment_config (dict)
        {
            "dataset": RetinaDataset,
            "model": "RetinaModel",
            "network": "resnet",
            "train_args": {
                "batch_size": 128
                "epochs": 10,
                "lr": 1e-3,
                "loss": "crossentropy",
                "optimizer": "adam",
            }
        }
    save_weights (bool)
        True => Save weights
    gpu_ind (int)
        specifies which gpu to use (or -1 for first available)
    use_wandb (boo)
        sync run to wandb
    """

    print(f"Running experiment with config {experiment_config} on GPU {gpu_ind}")

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {})
    }
    experiment_config["gpu_ind"] = gpu_ind

    datasets_module = importlib.import_module("core.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset = dataset_class_()
    dataset.load()
    print(dataset)

    def train_data_aug(img, lb):
        xs = []
        for _ in range(2):
            xs.append(train_classification_aug(img, lb, img_size=experiment_config["train_args"]['IMG_SIZE'])[0])
        img = tf.concat(xs, -1)
        return img, tf.one_hot(lb, 4)

    def val_data_aug(img, lb):
        xs = []
        for _ in range(2):
            xs.append(train_classification_aug(img, lb, img_size=experiment_config["train_args"]["IMG_SIZE"])[0])
        img = tf.concat(xs, -1)
        return img, tf.one_hot(lb, 4)

    def resize_only(img, lb):
        img = train_classification_aug(img, lb, img_size=experiment_config["train_args"]["IMG_SIZE"])[0]
        return img, tf.one_hot(lb, experiment_config["train_args"]['num_classes'])

    ds_train, ds_train_info = tfds.load('RetinaDataset', split='train', shuffle_files=True, as_supervised=True,
                                        with_info=True)
    ds_test = tfds.load('RetinaDataset', split='test', shuffle_files=True, as_supervised=True)
    num_examples = len(ds_train)

    if use_wandb:
        wandb.init(config=experiment_config)

    model = Pretrained_SimCLR_Model(num_classes=experiment_config["train_args"]['num_classes'],
                                    use_blur=experiment_config["train_args"]['use_blur'],
                                    proj_head_mode=experiment_config["train_args"]['proj_head_mode'],
                                    proj_out_dim=experiment_config["train_args"]['proj_out_dim'],
                                    num_proj_layers=experiment_config["train_args"]['num_proj_layers'],
                                    ft_proj_selector=experiment_config["train_args"]['ft_proj_selector'],
                                    resnet_depth=experiment_config["train_args"]['resnet_depth'],
                                    resnet_width_multiplier=experiment_config["train_args"]['resnet_width_multiplier'],
                                    resnet_se_ratio=experiment_config["train_args"]['resnet_se_ratio'],
                                    resnet_sk_ratio=experiment_config["train_args"]['resnet_sk_ratio'],
                                    hidden_norm=experiment_config["train_args"]['hidden_norm'],
                                    temperature=experiment_config["train_args"]['temperature'])
    # Build the Model
    input_shape_base = (None, experiment_config["train_args"]['IMG_SIZE'], experiment_config["train_args"]['IMG_SIZE'], 3)
    input_shape_simclr = (None, experiment_config["train_args"]['IMG_SIZE'], experiment_config["train_args"]['IMG_SIZE'], 6)
    model.base_model.build(input_shape_base)
    model.build(input_shape_simclr)

    lr_scheduler = WarmUpAndCosineDecay(experiment_config["train_args"]['learning_rate'],
                                        num_examples=num_examples,
                                        train_epochs=experiment_config["train_args"]['epochs'],
                                        train_batch_size=experiment_config["train_args"]['batch_size'],
                                        learning_rate_scaling=experiment_config["train_args"]['learning_rate_scaling'],
                                        warmup_epochs=experiment_config["train_args"]['warmup_epochs'])
    optimizer = tfa.optimizers.LAMB(lr_scheduler,
                                    weight_decay_rate=1e-6,
                                    exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])

    model.compile(optimizer=optimizer)

    model.fit(ds_train.map(train_data_aug).batch(experiment_config["train_args"]['batch_size']),
              epochs=experiment_config["train_args"]['epochs'],
              validation_data=ds_test.map(val_data_aug).batch(experiment_config["train_args"]['batch_size']),
              callbacks = [WandbCallback()])

    score = model.evaluate(ds_test.map(val_data_aug).batch(experiment_config["train_args"]['batch_size']))
    print(f"Test evaluation: {score}")

    if use_wandb:
        #wandb.log({"test_contrast_loss": round(score[0], 3)})
        #wandb.log({"contrast_acc": round(score[1], 3)})
        #wandb.log({"contrast_entropy": round(score[2], 3)})
        wandb.log({"test_loss": round(score[3], 3)})
        wandb.log({"test_acc": round(score[4], 3)})

    if save_weights:
        model.save_weights("simclr_pretrain_weights")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Provide index of GPU to use.")
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experiment JSON (\'{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnet"}\'',
    )
    parser.add_argument(
        "--nowandb", default=False, action="store_true", help="if true, do not use wandb"
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    if args.gpu < 0:
        gpu_manager = GPUManager()
        args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available
    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu, use_wandb=not args.nowandb)


if __name__ == "__main__":
    main()

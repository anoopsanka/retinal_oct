
import argparse
import importlib
import json
import os
from typing import Dict
from training.util import train_model
from training.gpu_manager import GPUManager
import pdb
import wandb

DEFAULT_TRAIN_ARGS = {"batch_size": 32, "epochs": 10, "lr": 1e-3}

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
                "lr": 1e-3
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
    print (dataset)


    models_module = importlib.import_module("core.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("core.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    
    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_
    )
    print (model)


    if use_wandb:
        wandb.init(config=experiment_config)

    train_model(
        model,
        dataset,
        epochs=experiment_config["train_args"]["epochs"],
        batch_size=experiment_config["train_args"]["batch_size"],
        lr = experiment_config["train_args"]["lr"],
        use_wandb=use_wandb
    )

    score = model.evaluate(dataset.test, batch_size=experiment_config["train_args"]["batch_size"])
    print(f"Test evaluation: {score}")

    if use_wandb:
        wandb.log({"test_metric": score})
    
    if save_weights:
        model.save_weights()


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

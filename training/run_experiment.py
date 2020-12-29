
import argparse
import importlib
import json
from typing import Dict
from training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size": 32, "epochs": 10}

def run_experiment(experiment_config: Dict, save_weights: bool):
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
                "epochs": 10
            }
        }
    save_weights (bool)
        True => Save weights
    """

    print(f"Running experiment with config {experiment_config}")

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {})
    }

    datasets_module = importlib.import_module("core.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    batch_size=experiment_config["train_args"]["batch_size"]
    dataset = dataset_class_(batch_size)
    dataset.load_or_generate_data()
    print (dataset)

    models_module = importlib.import_module("core.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("core.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    
    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_
    )
    print (model)


    train_model(
        model,
        dataset,
        epochs=experiment_config["train_args"]["epochs"],
    )

    score = model.evaluate(dataset.test)
    print(f"Test evaluation: {score}")
    
    if save_weights:
        model.save_weights()

def _parse_args():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)

if __name__ == "__main__":
    main()

#!/bin/bash
python training/run_experiment.py --save '{"dataset": "RetinaDatasetWrapper", "model": "RetinaModel", "network": "resnet", "train_args": {"batch_size": 32, "epochs": 1}}'
#python training/run_experiment.py --save '{"dataset": "RetinaDatasetWrapper", "model": "RetinaModel", "network": "resnetfinetune", "train_args": {"batch_size": 32, "epochs": 1}}'
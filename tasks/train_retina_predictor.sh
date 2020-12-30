#!/bin/bash
python training/run_experiment.py --save '{"dataset": "RetinaDatasetWrapper", "model": "RetinaModel", "network": "resnet", "train_args": {"batch_size": 32, "epochs": 1}}'
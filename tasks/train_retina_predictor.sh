#!/bin/bash
python training/run_experiment.py --save '{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnet", "train_args": {"batch_size": 32}}'
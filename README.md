# retinal_oct
Retinal OCT

## Context
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017).

Dataset and more details [here](https://www.kaggle.com/paultimothymooney/kermany2018)


## Pre-reqs
Complete [Setup](./setup.md)

Layout, tooling/code is based from https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project


## Training
You can run the shortcut command `tasks/train_retina_predictor.sh`, which runs the following:

```sh
python training/run_experiment.py --save '{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnetconv", "train_args": {"batch_size": 32}}'
```

## Running sweeps (hyper param optimization using weights & biases).
You can parallely run many sweeps, below is one example
```sh
wandb sweep training/sweep_resnet_finetune.yaml
copy the sweepid from the above command
wandb agent {sweepid}
```
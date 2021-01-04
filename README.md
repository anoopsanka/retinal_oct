# retinal_oct
Retinal OCT


## Pre-reqs
Complete [Setup](./setup.md)

Layout, tooling/code is based from https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project


## Training
You can run the shortcut command `tasks/train_retina_predictor.sh`, which runs the following:

```sh
python training/run_experiment.py --save '{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnet", "train_args": {"batch_size": 32}}'
```

## Running sweeps (hyper param optimization using weights & biases).
You can parallely run many sweeps, below is one example
```sh
wandb sweep training/sweep_resnet_finetune.yaml
copy the sweepid from the above command
wandb agent {sweepid}
```
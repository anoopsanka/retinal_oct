# retinal_oct
Retinal OCT


## Pre-reqs
Complete [Setup](./setup.md)

Layout, tooling is based from https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project


## Training
You can run the shortcut command `tasks/train_retina_predictor.sh`, which runs the following:

```sh
python training/run_experiment.py --save '{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnet", "train_args": {"batch_size": 32}}'
```
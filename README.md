# Retinal OCT

## Problem Statement
Identifying human diseases from medical images. Using supervised and semi-supervised techniques. Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017).

Resources

- [Introduction](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) 
- [Publication](https://bmcophthalmol.biomedcentral.com/articles/10.1186/s12886-020-01382-4)
- [Additional Information on the classification types](http://www.goodhopeeyeclinic.org.uk/oct.html)
- [Dataset](https://www.kaggle.com/paultimothymooney/kermany2018)



## Pre-reqs
Complete [Setup](./setup.md)


## Training
You can run the shortcut command `tasks/train_retina_predictor.sh`, which runs the following:

```sh
python training/run_experiment.py --save '{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnetconv", "train_args": {"batch_size": 32}}'
```

## Running sweeps (hyper param optimization using weights & biases).
You can parallely run many sweeps, below is one example
```sh
wandb sweep training/sweep_resnet_finetune.yaml
copy the sweepid from above
wandb agent {sweepid}
```

## Acknowledgements
- [Semi-Supervised Learning (SimClr)](https://github.com/google-research/simclr)
- [Full Stack Deeplearning](https://github.com/full-stack-deep-learning) 
- [Weights and Biases](https://wandb.ai/)
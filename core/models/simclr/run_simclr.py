import tensorflow_addons as tfa
import tensorflow as tf
import os
import random
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from importlib.util import find_spec
if find_spec("core") is None:
    import sys
    sys.path.append('..')

from core.datasets import RetinaDataset

from data_augmentation import preprocess_image, train_classification_aug
import simclr_model
import wandb
from wandb.keras import WandbCallback
from lr_schedule import WarmUpAndCosineDecay




# Load Retinal Data
ds_train, ds_train_info = tfds.load('RetinaDataset', split='train', shuffle_files=True, as_supervised=True, with_info=True)
ds_test  = tfds.load('RetinaDataset', split='test', shuffle_files=True, as_supervised=True)


# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

run = wandb.init(project='OCT-keras-SimCLR',
                 config={
                     # and include hyperparameters and metadata
                     "learning_rate": 0.01,
                     "epochs": 200,
                     "learning_rate_scaling": 'linear',
                     "num_examples": len(ds_train),
                     "warmup_epochs": 10,
                     "batch_size": 256,
                     "num_classes": 4,
                     "use_blur": True,
                     # ProjectionLayer Parameters
                     "proj_head_mode": 'nonlinear',
                     "proj_out_dim" : 128,
                     "num_proj_layers": 3, 
                     "ft_proj_selector": 0,
                     # Resnet_parameter
                     "resnet_depth": 18,
                     "resnet_width_multiplier": 1,
                     "resnet_se_ratio": 0.0,
                     "resnet_sk_ratio": 0.0,
                     # contrastive loss parameter
                     "hidden_norm": True,
                     "temperature" :1.0,
                     # Image Size
                     'IMG_SIZE': 128,
                 })


config = run.config

# Initialize the model
model = simclr_model.Pretrained_SimCLR_Model(num_classes=NUM_CLASSES,
                                             use_blur = config.use_blur,
                                             proj_head_mode = config.proj_head_mode,
                                             proj_out_dim = config.proj_out_dim,
                                             num_proj_layers = config.num_proj_layers,
                                             ft_proj_selector = config.ft_proj_selector,
                                             resnet_depth = config.resnet_depth,
                                             resnet_width_multiplier = config.resnet_width_multiplier,
                                             resnet_se_ratio = config.resnet_se_ratio,
                                             resnet_sk_ratio = config.resnet_sk_ratio,
                                             hidden_norm = config.hidden_norm,
                                             temperature = config.temperature)

# Build the Model
input_shape_base   = (None, config.IMG_SIZE, config.IMG_SIZE, 3)
input_shape_simclr = (None, config.IMG_SIZE, config.IMG_SIZE, 6)
model.base_model.build(input_shape_base)
model.build(input_shape_simclr)
model.summary()

# Define Scheduler and Optimizer
lr_scheduler = WarmUpAndCosineDecay(config.learning_rate, 
                                    num_examples= config.num_examples, 
                                    train_epochs= config.epochs, 
                                    train_batch_size = config.batch_size,
                                    learning_rate_scaling = config.learning_rate_scaling, 
                                    warmup_epochs = config.warmup_epochs)
optimizer = tfa.optimizers.LAMB(lr_scheduler,
                                weight_decay_rate=1e-6,
                                exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])

model.compile(optimizer= optimizer)


def train_data_aug(img, lb):
  xs = []
  for _ in range(2):
    xs.append(train_classification_aug(img, lb, img_size = config.IMG_SIZE)[0])
  img = tf.concat(xs, -1)
  return img, tf.one_hot(lb, config.num_classes)

def val_data_aug(img, lb):
  xs = []
  for _ in range(2):
    xs.append(train_classification_aug(img, lb, img_size = config.IMG_SIZE)[0])
  img = tf.concat(xs, -1)
  return img, tf.one_hot(lb, 4)

model.fit(ds_train.map(train_data_aug).batch(config.batch_size ),
          epochs= config.epochs,
          validation_data = ds_test.map(val_data_aug).batch(config.batch_size),
          callbacks = [WandbCallback()])

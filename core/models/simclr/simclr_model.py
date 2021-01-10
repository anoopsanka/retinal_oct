
import tensorflow as tf
from resnet import resnet
from projection_head import SupervisedHead, ProjectionHead

"""Wrapper for Resnet, Projection/ Supervised head to keras.Model
"""

class Pretrained_SimCLR_Model(tf.keras.Model):
  """Assemble the resnet, heads, training into one module"""
  def __init__(self, 
               num_classes=4,
               use_blur = True,
               # ProjectionLayer Parameters
               proj_head_mode='nonlinear',
               proj_out_dim = 128,
               num_proj_layers = 3, 
               ft_proj_selector = 0,
               # Resnet_parameter
               resnet_depth=18,
               resnet_width_multiplier=1,
               resnet_se_ratio = 0.0,
               resnet_sk_ratio = 0.0,
               # contrastive loss parameter
               hidden_norm      = True,
               temperature      = 1.0,
               **kwargs):
    
    
    super(Pretrained_SimCLR_Model, self).__init__(**kwargs)

    self.use_blur = use_blur

    # Defining the Base Resnet Model
    self.base_model = resnet(resnet_depth, 
                             resnet_width_multiplier, 
                             sk_ratio = resnet_se_ratio,
                             se_ratio = resnet_se_ratio)

    # Defining the Heads
    self.supervised_head  = SupervisedHead(num_classes=num_classes)
    self.projection_head  = ProjectionHead(proj_head_mode =proj_head_mode,
                                           proj_out_dim   = proj_out_dim,
                                           num_proj_layers = num_proj_layers)


    # Initialize metrics.
    all_metrics = []  # For summaries.
    self.contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
    self.contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
    self.contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')

    self.supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
    self.supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')

    all_metrics.extend([
        self.contrast_loss_metric, self.contrast_acc_metric, self.contrast_entropy_metric,
        self.supervised_loss_metric, self.supervised_acc_metric
    ])

    self.all_metrics = all_metrics

  
  def call(self, inputs, training=True):

    features = inputs
    if training:
      num_transforms = 2
    else:
      num_transforms = 1

    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(
        features, num_or_size_splits=num_transforms, axis=-1)
    
    # if self.use_blur and training:
    #   features_list = data_util.batch_random_blur(features_list,
    #                                               image_size,
    #                                               image_size)
    features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

    # run it through the base_model
    for layer in self.base_model.initial_conv_relu_max_pool:
      features = layer(features, training=training)

    for layer in self.base_model.block_groups:
      layer.trainable = False
      features = layer(features, training=training)
    
    # final average pool layer in Resnet
    features = tf.reduce_mean(features, [1, 2])

    # # Add heads
    projection_head_outputs, supervised_head_inputs = self.projection_head(features, training=training) 
    supervised_head_outputs  = self.supervised_head(tf.stop_gradient(supervised_head_inputs), training=training)

    return projection_head_outputs, supervised_head_outputs


  def train_step(self, data):
    X, y = data

    # taken from single_step!
    with tf.GradientTape() as tape:
      projection_head_outputs, supervised_head_outputs = self(X, training=True)
      loss = None

      # Evaluate Contrastive Loss
      outputs = projection_head_outputs
      con_loss, logits_con, labels_con = add_contrastive_loss(
          outputs,
          hidden_norm=hidden_norm,
          temperature=temperature
      )

      # Evaluate the Classification Loss
      outputs = supervised_head_outputs
      l = tf.concat([y, y], 0)
      sup_loss = add_supervised_loss(labels=l, logits=outputs)

      loss = tf.reduce_mean(con_loss + sup_loss)

      #TODO: add metrics updates here!
      # weight_decay = model_lib.add_weight_decay( 
      #     model, adjust_per_optimizer=True)
      # weight_decay_metric.update_state(weight_decay)
      # loss += weight_decay

      #total_loss_metric.update_state(loss)
      grads = tape.gradient(loss, self.trainable_variables)

      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

      # update the metrics
      update_pretrain_metrics_train(self.contrast_loss_metric,
                                    self.contrast_acc_metric,
                                    self.contrast_entropy_metric,
                                    con_loss, logits_con,
                                    labels_con)
      update_finetune_metrics_train(self.supervised_loss_metric,
                                    self.supervised_acc_metric, sup_loss, 
                                    l, outputs)
      
      return {m.name: m.result() for m in self.metrics}


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
  """Updated pretraining metrics."""
  contrast_loss.update_state(loss)

  contrast_acc_val = tf.equal(
      tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
  contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  contrast_acc.update_state(contrast_acc_val)

  prob_con = tf.nn.softmax(logits_con)
  entropy_con = -tf.reduce_mean(
      tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
  contrast_entropy.update_state(entropy_con)


def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
  supervised_loss_metric.update_state(loss)

  label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
  label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
  supervised_acc_metric.update_state(label_acc)
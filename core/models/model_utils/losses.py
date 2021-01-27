import tensorflow as tf


# trim down version of the actual TPU implementation
# where hidden1_large is a BANK of hidden vectors from different batch
def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
    """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (2*bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector
    weights: a weighting number or vector
  
  Returns:
    A loss scalar
    The logits for contrastive prediction task
  """

    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)

    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    LARGE_NUM = 1e9

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1), )
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1), )
    loss = loss_a + loss_b

    return loss, logits_ab, labels


def add_supervised_loss(labels, logits):
    """Compute mean supervised loss over local batch."""
    losses = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                    logits)
    return tf.reduce_mean(losses)

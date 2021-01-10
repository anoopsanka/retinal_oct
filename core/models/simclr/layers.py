
import tensorflow as tf
"""Contains definitions for the post-activation form of Residual Networks.
Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385


this has been adopted to NOT use global flag declarations (absl flags)
"""


class BatchNormRelu(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               relu=True,
               init_zero=False,
               center=True,
               scale=True,
               data_format='channels_last',
               global_bn=False,
               batch_norm_decay = False,
               BATCH_NORM_EPSILON = 1e-5,
               **kwargs):
    super(BatchNormRelu, self).__init__(**kwargs)
    self.relu = relu
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()
    if data_format == 'channels_first':
      axis = 1
    else:
      axis = -1
    if global_bn:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
          axis=axis,
          momentum=batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          gamma_initializer=gamma_initializer)
    else:
      # TODO(srbs): Set fused=True
      # Batch normalization layers with fused=True only support 4D input
      # tensors.
      self.bn = tf.keras.layers.BatchNormalization(
          axis=axis,
          momentum=batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          fused=False,
          gamma_initializer=gamma_initializer)

  def call(self, inputs, training):
    inputs = self.bn(inputs, training=training)
    if self.relu:
      inputs = tf.nn.relu(inputs)
    return inputs


class DropBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               keep_prob,
               dropblock_size,
               data_format='channels_last',
               **kwargs):
    self.keep_prob = keep_prob
    self.dropblock_size = dropblock_size
    self.data_format = data_format
    super(DropBlock, self).__init__(**kwargs)

  def call(self, net, training):
    keep_prob = self.keep_prob
    dropblock_size = self.dropblock_size
    data_format = self.data_format
    if not training or keep_prob is None:
      return net

    tf.logging.info(
        'Applying DropBlock: dropblock_size {}, net.shape {}'.format(
            dropblock_size, net.shape))

    if data_format == 'channels_last':
      _, width, height, _ = net.get_shape().as_list()
    else:
      _, _, width, height = net.get_shape().as_list()
    if width != height:
      raise ValueError('Input tensor with width!=height is not supported.')

    dropblock_size = min(dropblock_size, width)
    # seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
        width - dropblock_size + 1)**2

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
    valid_block_center = tf.logical_and(
        tf.logical_and(w_i >= int(dropblock_size // 2),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= int(dropblock_size // 2),
                       h_i < width - (dropblock_size - 1) // 2))

    valid_block_center = tf.expand_dims(valid_block_center, 0)
    valid_block_center = tf.expand_dims(
        valid_block_center, -1 if data_format == 'channels_last' else 0)

    randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
    block_pattern = (
        1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
            (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if dropblock_size == width:
      block_pattern = tf.reduce_min(
          block_pattern,
          axis=[1, 2] if data_format == 'channels_last' else [2, 3],
          keepdims=True)
    else:
      if data_format == 'channels_last':
        ksize = [1, dropblock_size, dropblock_size, 1]
      else:
        ksize = [1, 1, dropblock_size, dropblock_size]
      block_pattern = -tf.nn.max_pool(
          -block_pattern,
          ksize=ksize,
          strides=[1, 1, 1, 1],
          padding='SAME',
          data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

    percent_ones = (
        tf.cast(tf.reduce_sum((block_pattern)), tf.float32) /
        tf.cast(tf.size(block_pattern), tf.float32))

    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net


class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self, kernel_size, data_format='channels_last', **kwargs):
    super(FixedPadding, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs, training):
    kernel_size = self.kernel_size
    data_format = self.data_format
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
      padded_inputs = tf.pad(
          inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               filters,
               kernel_size,
               strides,
               data_format='channels_last',
               **kwargs):
    super(Conv2dFixedPadding, self).__init__(**kwargs)
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
    else:
      self.fixed_padding = None
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        data_format=data_format)

  def call(self, inputs, training):
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs, training=training)


class IdentityLayer(tf.keras.layers.Layer):

  def call(self, inputs, training):
    return tf.identity(inputs)


class SK_Conv2D(tf.keras.layers.Layer):  # pylint: disable=invalid-name
  """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

  def __init__(self,
               filters,
               strides,
               sk_ratio,
               min_dim=32,
               data_format='channels_last',
               **kwargs):
    super(SK_Conv2D, self).__init__(**kwargs)
    self.data_format = data_format
    self.filters = filters
    self.sk_ratio = sk_ratio
    self.min_dim = min_dim

    # Two stream convs (using split and both are 3x3).
    self.conv2d_fixed_padding = Conv2dFixedPadding(
        filters=2 * filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)
    self.batch_norm_relu = BatchNormRelu(data_format=data_format)

    # Mixing weights for two streams.
    mid_dim = max(int(filters * sk_ratio), min_dim)
    self.conv2d_0 = tf.keras.layers.Conv2D(
        filters=mid_dim,
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        use_bias=False,
        data_format=data_format)
    self.batch_norm_relu_1 = BatchNormRelu(data_format=data_format)
    self.conv2d_1 = tf.keras.layers.Conv2D(
        filters=2 * filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        use_bias=False,
        data_format=data_format)

  def call(self, inputs, training):
    channel_axis = 1 if self.data_format == 'channels_first' else 3
    pooling_axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

    # Two stream convs (using split and both are 3x3).
    inputs = self.conv2d_fixed_padding(inputs, training=training)
    inputs = self.batch_norm_relu(inputs, training=training)
    inputs = tf.stack(tf.split(inputs, num_or_size_splits=2, axis=channel_axis))

    # Mixing weights for two streams.
    global_features = tf.reduce_mean(
        tf.reduce_sum(inputs, axis=0), pooling_axes, keepdims=True)
    global_features = self.conv2d_0(global_features, training=training)
    global_features = self.batch_norm_relu_1(global_features, training=training)
    mixing = self.conv2d_1(global_features, training=training)
    mixing = tf.stack(tf.split(mixing, num_or_size_splits=2, axis=channel_axis))
    mixing = tf.nn.softmax(mixing, axis=0)

    return tf.reduce_sum(inputs * mixing, axis=0)


class SE_Layer(tf.keras.layers.Layer):  # pylint: disable=invalid-name
  """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""

  def __init__(self, filters, se_ratio, data_format='channels_last', **kwargs):
    super(SE_Layer, self).__init__(**kwargs)
    self.data_format = data_format
    self.se_reduce = tf.keras.layers.Conv2D(
        max(1, int(filters * se_ratio)),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        padding='same',
        data_format=data_format,
        use_bias=True)
    self.se_expand = tf.keras.layers.Conv2D(
        None,  # This is filled later in build().
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        padding='same',
        data_format=data_format,
        use_bias=True)

  def build(self, input_shape):
    self.se_expand.filters = input_shape[-1]
    super(SE_Layer, self).build(input_shape)

  def call(self, inputs, training):
    spatial_dims = [2, 3] if self.data_format == 'channels_first' else [1, 2]
    se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
    se_tensor = self.se_expand(tf.nn.relu(self.se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * inputs
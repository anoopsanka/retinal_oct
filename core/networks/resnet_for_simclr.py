from core.models.model_utils.resnet_blocks import *


class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 block_fn,
                 blocks,
                 strides,
                 data_format='channels_last',
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):
        self._name = kwargs.get('name')
        self.sk_ratio = sk_ratio
        self.se_ratio = se_ratio
        super(BlockGroup, self).__init__(**kwargs)

        self.layers = []
        self.layers.append(
            block_fn(
                filters,
                strides,
                use_projection=True,
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
                sk_ratio=sk_ratio,
                se_ratio=se_ratio))

        for _ in range(1, blocks):
            self.layers.append(
                block_fn(
                    filters,
                    1,
                    data_format=data_format,
                    dropblock_keep_prob=dropblock_keep_prob,
                    dropblock_size=dropblock_size,
                    sk_ratio=sk_ratio,
                    se_ratio=se_ratio))

    def call(self, inputs, training):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return tf.identity(inputs, self._name)


class Resnet(tf.keras.Model):
    """Define base resnet layer"""

    def __init__(self,
                 block_fn,
                 layers,
                 width_multiplier,
                 cifar_stem=False,
                 data_format='channels_last',
                 dropblock_keep_probs=None,
                 dropblock_size=None,
                 trainable=True,
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.data_format = data_format
        if dropblock_keep_probs is None:
            dropblock_keep_probs = [None] * 4
        if not isinstance(dropblock_keep_probs,
                          list) or len(dropblock_keep_probs) != 4:
            raise ValueError('dropblock_keep_probs is not valid:',
                             dropblock_keep_probs)

        self.initial_conv_relu_max_pool = []
        if cifar_stem:
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier,
                    kernel_size=3,
                    strides=1,
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_conv', trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_max_pool', trainable=trainable))
        else:
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        filters=64 * width_multiplier // 2,
                        kernel_size=3,
                        strides=2,
                        data_format=data_format,
                        trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    BatchNormRelu(data_format=data_format, trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        filters=64 * width_multiplier // 2,
                        kernel_size=3,
                        strides=1,
                        data_format=data_format,
                        trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    BatchNormRelu(data_format=data_format, trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        filters=64 * width_multiplier,
                        kernel_size=3,
                        strides=1,
                        data_format=data_format,
                        trainable=trainable))
            else:
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        filters=64 * width_multiplier,
                        kernel_size=7,
                        strides=2,
                        data_format=data_format,
                        trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_conv', trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))

            self.initial_conv_relu_max_pool.append(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='SAME',
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_max_pool', trainable=trainable))

        self.block_groups = []
        # TODO(srbs): This impl is different from the original one in the case where
        # fine_tune_after_block != 4. In that case earlier BN stats were getting
        # updated. Now they will not be. Check with Ting to make sure this is ok.

        self.block_groups.append(
            BlockGroup(
                filters=64 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[0],
                strides=1,
                name='block_group1',
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_probs[0],
                dropblock_size=dropblock_size,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

        self.block_groups.append(
            BlockGroup(
                filters=128 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[1],
                strides=2,
                name='block_group2',
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_probs[1],
                dropblock_size=dropblock_size,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

        self.block_groups.append(
            BlockGroup(
                filters=256 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[2],
                strides=2,
                name='block_group3',
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_probs[2],
                dropblock_size=dropblock_size,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

        self.block_groups.append(
            BlockGroup(
                filters=512 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[3],
                strides=2,
                name='block_group4',
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_probs[3],
                dropblock_size=dropblock_size,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

    def call(self, inputs, training):
        for layer in self.initial_conv_relu_max_pool:
            inputs = layer(inputs, training=training)

        for i, layer in enumerate(self.block_groups):
            inputs = layer(inputs, training=training)

        inputs = tf.identity(inputs, 'final_conv_block')

        # if self.data_format == 'channels_last':
        #   inputs = tf.reduce_mean(inputs, [1, 2])
        # else:
        #   inputs = tf.reduce_mean(inputs, [2, 3])

        inputs = tf.identity(inputs, 'final_avg_pool')
        return inputs


def resnet(resnet_depth,
           width_multiplier,
           cifar_stem=False,
           data_format='channels_last',
           dropblock_keep_probs=None,
           dropblock_size=None,
           sk_ratio=0.0,
           se_ratio=0.0):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {
            'block': ResidualBlock,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': ResidualBlock,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': BottleneckBlock,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': BottleneckBlock,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': BottleneckBlock,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': BottleneckBlock,
            'layers': [3, 24, 36, 3]
        }
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return Resnet(
        params['block'],
        params['layers'],
        width_multiplier,
        cifar_stem=cifar_stem,
        dropblock_keep_probs=dropblock_keep_probs,
        dropblock_size=dropblock_size,
        data_format=data_format,
        sk_ratio=sk_ratio,
        se_ratio=se_ratio)

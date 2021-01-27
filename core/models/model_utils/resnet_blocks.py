from core.models.model_utils.layers import *


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 data_format='channels_last',
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        del dropblock_keep_prob
        del dropblock_size
        self.conv2d_bn_layers = []
        self.shortcut_layers = []
        if use_projection:
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                if strides > 1:
                    self.shortcut_layers.append(FixedPadding(2, data_format))
                self.shortcut_layers.append(
                    tf.keras.layers.AveragePooling2D(
                        pool_size=2,
                        strides=strides,
                        padding='SAME' if strides == 1 else 'VALID',
                        data_format=data_format))
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        filters=filters,
                        kernel_size=1,
                        strides=1,
                        data_format=data_format))
            else:
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        filters=filters,
                        kernel_size=1,
                        strides=strides,
                        data_format=data_format))
            self.shortcut_layers.append(
                BatchNormRelu(relu=False, data_format=data_format))

        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=3,
                strides=strides,
                data_format=data_format))
        self.conv2d_bn_layers.append(BatchNormRelu(data_format=data_format))
        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=3, strides=1, data_format=data_format))
        self.conv2d_bn_layers.append(
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format))
        if se_ratio > 0:
            self.se_layer = SE_Layer(filters, se_ratio, data_format=data_format)

        self.se_ratio = se_ratio

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.shortcut_layers:
            # Projection shortcut in first layer to match filters and strides
            shortcut = layer(shortcut, training=training)

        for layer in self.conv2d_bn_layers:
            inputs = layer(inputs, training=training)

        if self.se_ratio > 0:
            inputs = self.se_layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
    """BottleneckBlock."""

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 data_format='channels_last',
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.projection_layers = []
        if use_projection:
            filters_out = 4 * filters
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                if strides > 1:
                    self.projection_layers.append(FixedPadding(2, data_format))
                self.projection_layers.append(
                    tf.keras.layers.AveragePooling2D(
                        pool_size=2,
                        strides=strides,
                        padding='SAME' if strides == 1 else 'VALID',
                        data_format=data_format))
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        filters=filters_out,
                        kernel_size=1,
                        strides=1,
                        data_format=data_format))
            else:
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        filters=filters_out,
                        kernel_size=1,
                        strides=strides,
                        data_format=data_format))
            self.projection_layers.append(
                BatchNormRelu(relu=False, data_format=data_format))
        self.shortcut_dropblock = DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size)

        self.conv_relu_dropblock_layers = []

        self.conv_relu_dropblock_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=1, strides=1, data_format=data_format))
        self.conv_relu_dropblock_layers.append(
            BatchNormRelu(data_format=data_format))
        self.conv_relu_dropblock_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size))

        if sk_ratio > 0:
            self.conv_relu_dropblock_layers.append(
                SK_Conv2D(filters, strides, sk_ratio, data_format=data_format))
        else:
            self.conv_relu_dropblock_layers.append(
                Conv2dFixedPadding(
                    filters=filters,
                    kernel_size=3,
                    strides=strides,
                    data_format=data_format))
            self.conv_relu_dropblock_layers.append(
                BatchNormRelu(data_format=data_format))
        self.conv_relu_dropblock_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size))

        self.conv_relu_dropblock_layers.append(
            Conv2dFixedPadding(
                filters=4 * filters,
                kernel_size=1,
                strides=1,
                data_format=data_format))
        self.conv_relu_dropblock_layers.append(
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format))
        self.conv_relu_dropblock_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size))

        if se_ratio > 0:
            self.conv_relu_dropblock_layers.append(
                SE_Layer(filters, se_ratio, data_format=data_format))

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.projection_layers:
            shortcut = layer(shortcut, training=training)
        shortcut = self.shortcut_dropblock(shortcut, training=training)

        for layer in self.conv_relu_dropblock_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)

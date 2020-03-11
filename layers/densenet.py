import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
"""


def PreprocessBlock():
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3))),
        tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False),
        tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.MaxPooling2D(3, strides=2)
    ])


class _ConvBlock(CustomLayer):

    def __init__(self, growth_rate):
        self.growth_rate = growth_rate
        super().__init__(growth_rate=growth_rate)

    def build(self, input_shape):
        self.seq = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(4 * self.growth_rate, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.growth_rate, 3,
                                   padding='same', use_bias=False)
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([inputs, self.seq(inputs)])


def DenseBlock(blocks):
    return tf.keras.Sequential([
        _ConvBlock(growth_rate=32) for _ in range(blocks)
    ])


class TransitionBlock(CustomLayer):
    def __init__(self, reduction):
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def build(self, input_shape):

        self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            int(input_shape[-1] * self.reduction), 1, use_bias=False)
        self.pool = tf.keras.layers.AveragePooling2D(2, strides=2)

    def call(self, inputs):
        x = inputs
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


def DenseNet(blocks):
    return tf.keras.Sequential([
        PreprocessBlock(),
        DenseBlock(blocks=blocks[0]),
        TransitionBlock(reduction=0.5),
        DenseBlock(blocks=blocks[1]),
        TransitionBlock(reduction=0.5),
        DenseBlock(blocks=blocks[2]),
        TransitionBlock(reduction=0.5),
        DenseBlock(blocks=blocks[3]),
        tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
        tf.keras.layers.ReLU()
    ])


def DenseNet121():
    return DenseNet(blocks=[6, 12, 24, 16])


def DenseNet169():
    return DenseNet(blocks=[6, 12, 32, 32])


def DenseNet201():
    return DenseNet(blocks=[6, 12, 48, 32])

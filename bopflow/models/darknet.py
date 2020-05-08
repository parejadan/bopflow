import tensorflow as tf
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    LeakyReLU,
    Add,
    Input,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.regularizers import l2

from bopflow.models.batch_norm import BatchNormalization


def darknet_conv(x, filters: int, size: int, strides=1, batch_norm=True):
    if strides == 1:
        padding_flag = "same"
    else:  # do top left half-padding
        padding_flag = "valid"
        padding = ((1, 0), (1, 0))
        x = ZeroPadding2D(padding)(x)

    use_bias = not batch_norm
    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding_flag,
        use_bias=use_bias,
        kernel_regularizer=l2(0.0005),
    )(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def darknet_residual(x, filters: int):
    prev_layer = x
    x = darknet_conv(x=x, filters=filters // 2, size=1)
    x = darknet_conv(x=x, filters=filters, size=3)
    x = Add()([prev_layer, x])

    return x


def darknet_block(x, filters: int, blocks: int):
    x = darknet_conv(x=x, filters=filters, size=3, strides=2)
    for _ in range(blocks):
        x = darknet_residual(x=x, filters=filters)

    return x


def darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x=x, filters=32, size=3)
    x = darknet_block(x=x, filters=64, blocks=1)
    x = darknet_block(x=x, filters=128, blocks=2)
    x = x_36 = darknet_block(x=x, filters=256, blocks=8)
    x = x_61 = darknet_block(x=x, filters=512, blocks=8)
    x = darknet_block(x=x, filters=1024, blocks=4)

    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def darknet_conv_upsampling(x_in: tuple, filters: int, size: int, up_sampling: int):
    inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
    x, x_skip = inputs

    # concat with skip connection
    x = darknet_conv(x=x, filters=filters, size=size)
    x = UpSampling2D(up_sampling)(x)
    x = Concatenate()([x, x_skip])

    return x, inputs

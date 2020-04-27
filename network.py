import tensorflow as tf
import logging
import numpy as np 
from tensorflow.keras import layers


def channel_padding(x):
    """
    zero padding in an axis of channel 
    """

    return tf.keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)


def singleBlazeBlock(x, filters=24, kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution
    x_0 = layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = layers.BatchNormalization()(x_0) #layers.BatchNormalization

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]

        x_ = layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = layers.Lambda(channel_padding)(x_)

        out = layers.Add()([x_1, x_])
        return layers.Activation("relu")(out)

    out = layers.Add()([x_1, x])
    return layers.Activation("relu")(out)


def doubleBlazeBlock(x, filters_1=24, filters_2=96, kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution, project
    x_0 = layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = layers.BatchNormalization()(x_0)

    x_2 = layers.Activation("relu")(x_1)

    # depth-wise separable convolution, expand
    x_3 = layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)

    x_4 = layers.BatchNormalization()(x_3)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]

        x_ = layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = layers.Lambda(channel_padding)(x_)

        out = layers.Add()([x_4, x_])
        return layers.Activation("relu")(out)

    out = layers.Add()([x_4, x])
    return layers.Activation("relu")(out)


def feature_extractor(input_shape):

    inputs = layers.Input(shape=input_shape)

    x_0 = layers.Conv2D(filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = layers.BatchNormalization()(x_0)
    x_0 = layers.Activation("relu")(x_0)

    # single BlazeBlock phase
    x_1 = singleBlazeBlock(x_0)
    x_2 = singleBlazeBlock(x_1)
    x_3 = singleBlazeBlock(x_2, strides=2, filters=48)
    x_4 = singleBlazeBlock(x_3, filters=48)
    x_5 = singleBlazeBlock(x_4, filters=48)

    # double BlazeBlock phase
    x_6 = doubleBlazeBlock(x_5, strides=2)
    x_7 = doubleBlazeBlock(x_6)
    x_8 = doubleBlazeBlock(x_7)
    x_9 = doubleBlazeBlock(x_8, strides=2)
    x10 = doubleBlazeBlock(x_9)
    x11 = doubleBlazeBlock(x10)

    model = tf.keras.models.Model(inputs=inputs, outputs=[x_8, x11])

    return model


def blaze_face_detector(input_shape, n_boxes=[2, 6], n_classes=3):

    base = feature_extractor(input_shape) # (224, 224, 3)

    # confidence
    bb_16_conf = layers.Conv2D(filters=n_boxes[0] * n_classes, kernel_size=3, padding='same', activation='sigmoid')(base.output[0])
    bb_16_conf = layers.Reshape((16**2 * n_boxes[0], n_classes))(bb_16_conf)

    bb_8_conf  = layers.Conv2D(filters=n_boxes[1] * n_classes, kernel_size=3, padding='same', activation='sigmoid')(base.output[1])
    bb_8_conf  = layers.Reshape((8**2 * n_boxes[1], n_classes))(bb_8_conf)

    # Concatenate prediction - shape : [batch_size, 896, n_classes]
    conf_of_bb = layers.Concatenate(axis=1, name='clf_output')([bb_16_conf, bb_8_conf])

    # location  [x, y, w, h]
    bb_16_loc = layers.Conv2D(filters=n_boxes[0] * 4, kernel_size=3, padding='same')(base.output[0])
    bb_16_loc = layers.Reshape((16**2 * n_boxes[0], 4))(bb_16_loc)

    bb_8_loc  = layers.Conv2D(filters=n_boxes[1] * 4, kernel_size=3, padding='same')(base.output[1])
    bb_8_loc  = layers.Reshape((8**2 * n_boxes[1], 4))(bb_8_loc)

    # Concatenate prediction - shape : [batch_size, 896, n_classes]
    loc_of_bb = layers.Concatenate(axis=1)([bb_16_loc, bb_8_loc])

    # output_combined = layers.Concatenate(axis=-1, name='bb_output')([conf_of_bb, loc_of_bb])

    # Detectors model
    # return tf.keras.models.Model(base.input, [conf_of_bb, output_combined])
    return tf.keras.models.Model(base.input, [conf_of_bb, loc_of_bb])


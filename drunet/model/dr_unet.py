import os
import pathlib

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import *
from tensorflow.keras.layers import *


def BN(inputs, training=True):
    bn = keras.layers.BatchNormalization()
    return bn(inputs, training=training)


def conv_block(input_tensor, num_filters, kernel_size, relu=False):
    # 卷积核尺寸为1的卷积层
    encoder = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same')(input_tensor)
    encoder = BN(encoder)
    if relu:
        encoder = tf.keras.layers.ReLU()(encoder)
    else:
        encoder = tf.keras.layers.ELU()(encoder)
    return encoder


def conv_13_block(input_tensor, num_filters, filter_size, decrease=True, relu=False):
    if decrease:
        num_filters = num_filters // 4
    output = conv_block(input_tensor, num_filters, kernel_size=filter_size[0], relu=relu)
    output = conv_block(output, num_filters, kernel_size=filter_size[1], relu=relu)
    return output


def conv_decrease_block(input_tensor, num_filters, filter_size, decrease=False, residual=False, relu=False):
    # 传统的UNet三层卷积全部filter_size=3, 但是本文对每层卷积核尺寸进行改变
    # 先使用1x1的卷积核进行降维, 随后使用3x3的卷积核，最后再用1x1的卷积核，但是卷积核的数量不变
    # 并且在输出时，采用残差的方式，将输入与输出相加后在返回
    if decrease:
        num_filters = num_filters // 4
    output = conv_block(input_tensor, num_filters, kernel_size=filter_size[0], relu=relu)
    output = conv_block(output, num_filters, kernel_size=filter_size[1], relu=relu)
    output = conv_block(output, num_filters, kernel_size=filter_size[2], relu=relu)
    if residual:
        output = output + conv_block(input_tensor, num_filters, kernel_size=1, relu=relu)
    return output


def conv_increase_block(input_tensor, num_filters, filter_size, increase=False, residual=False, relu=False):
    # 仍然采用131卷积层叠加的方式，但是在最后一层将卷积核的数量放大四倍。
    if increase:
        num_filters = num_filters // 4
        final_filters = num_filters * 4
    else:
        final_filters = num_filters
    output = conv_block(input_tensor, num_filters, kernel_size=filter_size[0], relu=relu)
    output = conv_block(output, num_filters, kernel_size=filter_size[1], relu=relu)
    output = conv_block(output, final_filters, kernel_size=filter_size[2], relu=relu)
    if residual:
        output = output + conv_block(input_tensor, final_filters, kernel_size=1, relu=relu)
    return output


def encoder_block(input_tensor, num_filters, filter_size=None, increase=True,
                  decrease=True, residual=True, pool=True, relu=False, drop_rate=0.2):
    if filter_size is None:
        filter_size = [1, 3, 3]
    encoder = conv_decrease_block(input_tensor, num_filters, filter_size, decrease, residual, relu)
    encoder = tf.keras.layers.Dropout(drop_rate)(encoder)
    encoder = conv_increase_block(encoder, num_filters, filter_size, increase, residual, relu)
    if pool:
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder
    else:
        return encoder


def decoder_block(input_tensor, concat_tensor, num_filters, filter_size=None,
                  increase=True, decrease=True, residual=True, relu=False, drop_rate=0.2):
    if filter_size is None:
        filter_size = [1, 3, 3]
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = BN(decoder)
    if relu:
        decoder = Activation('relu')(decoder)
    else:
        decoder = ELU()(decoder)
    decoder = conv_decrease_block(decoder, num_filters, filter_size, decrease, residual, relu)
    decoder = tf.keras.layers.Dropout(drop_rate)(decoder)
    decoder = conv_increase_block(decoder, num_filters, filter_size, increase, residual, relu)
    return decoder


def encoder_block_131(input_tensor, num_filters, filter_size=None, increase=True,
                      decrease=True, residual=True, pool=True, relu=False, drop_rate=0.2):
    if filter_size is None:
        filter_size = [1, 3, 1]
    encoder = conv_13_block(input_tensor, num_filters, filter_size[:2], decrease, relu)
    encoder = tf.keras.layers.Dropout(drop_rate)(encoder)
    encoder = conv_increase_block(encoder, num_filters, filter_size, increase, residual, relu)
    if pool:
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder
    else:
        return encoder


def decoder_block_131(input_tensor, concat_tensor, num_filters, filter_size=None,
                      increase=True, decrease=True, residual=True, relu=False, drop_rate=0.2):
    if filter_size is None:
        filter_size = [1, 3, 1]
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = BN(decoder)
    if relu:
        decoder = Activation('relu')(decoder)
    else:
        decoder = ELU()(decoder)
    decoder = conv_13_block(decoder, num_filters, filter_size[:2], decrease, relu)
    decoder = tf.keras.layers.Dropout(drop_rate)(decoder)
    decoder = conv_increase_block(decoder, num_filters, filter_size, increase, residual, relu)
    return decoder


def my_model(input_shape, dim=32):
    inputs = keras.Input(shape=input_shape)

    inputs_1 = conv_block(inputs, dim // 2, kernel_size=1)
    # 编码层， encoder0 经过卷积但是图像尺寸没变， encoder1缩小一半
    encoder0_pool, encoder0 = encoder_block_131(inputs_1, dim)  # 1
    encoder1_pool, encoder1 = encoder_block_131(encoder0_pool, dim * 2)  # 1/2
    encoder2_pool, encoder2 = encoder_block_131(encoder1_pool, dim * 4)  # 1/4
    encoder3_pool, encoder3 = encoder_block_131(encoder2_pool, dim * 8)  # 1/8
    encoder4_pool, encoder4 = encoder_block_131(encoder3_pool, dim * 16)  # 1/16

    center = encoder_block_131(encoder4_pool, dim * 32, pool=False)  # 原始图像的1/16

    # 解码层
    decoder4 = decoder_block_131(center, encoder4, dim * 16)
    decoder3 = decoder_block_131(decoder4, encoder3, dim * 8)
    decoder2 = decoder_block_131(decoder3, encoder2, dim * 4)
    decoder1 = decoder_block_131(decoder2, encoder1, dim * 2)
    decoder0 = decoder_block_131(decoder1, encoder0, dim)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = keras.Model(inputs, outputs, name='my_model_1')
    return model


if __name__ == '__main__':
    seg_model = my_model(input_shape=(256, 256, 1))
    seg_model.summary()

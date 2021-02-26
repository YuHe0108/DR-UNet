import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_package.Module import CBAM

"""单一尺度连接"""


def conv_same_layer(inputs, filters, k_size=3, stride=1, dilate_rate=1, use_bias=False):
    """
    不该变特征图的大小的卷积层 (当rate>1的时候，为扩张卷积)
    """
    if stride == 1:
        return layers.Conv2D(filters,
                             k_size,
                             stride,
                             padding='same',
                             use_bias=use_bias,
                             dilation_rate=dilate_rate,
                             kernel_initializer=keras.initializers.he_normal())(inputs)
    else:
        k_size_effective = k_size + (k_size - 1) * (dilate_rate - 1)
        pad_total = k_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = layers.ZeroPadding2D((pad_beg, pad_end))(inputs)
        outputs = layers.Conv2D(filters,
                                k_size,
                                stride,
                                padding='valid',
                                use_bias=use_bias,
                                dilation_rate=dilate_rate,
                                kernel_initializer=keras.initializers.he_normal())(inputs)
        return outputs


def conv_block(inputs,
               filters,
               kernel_size,
               strides=1,
               bn=True,
               activation='relu',
               dilate_rate=1,
               cbam=False):
    """conv+bn+relu"""
    out = conv_same_layer(
        inputs, filters, kernel_size, strides, dilate_rate=dilate_rate, use_bias=not bn)
    if bn:
        out = layers.BatchNormalization()(out)
    if activation != 'none':
        out = layers.Activation(activation)(out)
    if cbam:
        c_out = layers.Multiply()([CBAM.channel_attention_conv(out, filters), out])
        out = layers.Multiply()([CBAM.SpatialAttention()(c_out), c_out])
    return out


def residual_block(inputs, filters, kernel_size, activation='relu', rate=1, cbam=False):
    out = conv_block(inputs, filters // 2, kernel_size=1)
    out = conv_block(out, filters // 2, kernel_size, dilate_rate=rate)

    # channel attention  + spatial attention
    if cbam:
        c_out = layers.Multiply()([CBAM.channel_attention_conv(out, filters // 2), out])
        out = layers.Multiply()([CBAM.SpatialAttention()(c_out), c_out])

    # 输出卷积层
    out = conv_block(out, filters, kernel_size=1, activation='none')
    if inputs.shape[-1] != filters:
        short_cut = conv_block(inputs, filters, kernel_size=1, activation='none')
        out = layers.Add()([short_cut, out])
    else:
        out = layers.Add()([inputs, out])
    out = layers.Activation(activation)(out)
    return out


def muti_conv_residual_block(inputs,
                             out_filters,
                             conv_filters,
                             cbam=True,
                             residual=True,
                             layer_per_block=3,
                             dilate_rate=1):
    outputs = [inputs]
    x = inputs
    for i in range(layer_per_block):
        x = conv_block(x, conv_filters, kernel_size=3, dilate_rate=dilate_rate)
        outputs.append(x)

    # 特征图拼接
    concat_out = layers.Concatenate()(outputs)
    # 整合拼接的特征图
    out = conv_block(concat_out, filters=out_filters, kernel_size=1)
    if cbam:
        # 使用空间和通道注意力集中
        c_out = layers.Multiply()([CBAM.channel_attention_conv(out, out_filters), out])
        out = layers.Multiply()([CBAM.SpatialAttention()(c_out), c_out])
    if residual:
        # 是否使用残差结构
        out = layers.Add()([inputs, out])
    return out


def my_model(input_shape, num_classes=1, dims=32, kernel_size=3):
    print(dims)
    conv_dims_list = [32, 64, 96, 128, 160, 192, 224, 320, 416, 512]
    out_dims_list = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280]

    inputs = layers.Input(input_shape)
    # ------------------------------ 下采样层------------------------------------
    # 初始层: out_2: 下采样2倍
    out_2 = conv_block(inputs, out_dims_list[0], kernel_size=3, strides=2)
    out_2_r = muti_conv_residual_block(
        out_2, out_dims_list[0], conv_dims_list[0], cbam=False)  # [128, 128, 128]

    # out_4: 下采样4倍
    out_4 = conv_block(out_2_r, out_dims_list[1], kernel_size, strides=2)
    out_4_r = muti_conv_residual_block(
        out_4, out_dims_list[1], conv_dims_list[1], cbam=False)  # [64, 64, 256]

    # out_8: 下采样8倍
    out_8 = conv_block(out_4_r, out_dims_list[2], kernel_size, strides=2)
    out_8_r = muti_conv_residual_block(
        out_8, out_dims_list[2], conv_dims_list[2], cbam=False)  # [32, 32, 512]

    # out_16: 下采样16倍
    out_16 = conv_block(out_8_r, out_dims_list[3], kernel_size, strides=2)
    out_16_r = muti_conv_residual_block(
        out_16, out_dims_list[3], conv_dims_list[3], cbam=False)  # [16, 16, 768]

    # out_32: 下采样32倍
    out_32 = conv_block(out_16_r, out_dims_list[4], kernel_size, strides=2)
    out_32_r = muti_conv_residual_block(
        out_32, out_dims_list[4], conv_dims_list[4], cbam=False)  # [8, 8, 1024]

    # ------------------------------ 中间层------------------------------------
    center_32 = conv_block(out_32_r, out_dims_list[4], kernel_size, strides=1)
    center_32 = muti_conv_residual_block(
        center_32, out_dims_list[4], conv_dims_list[4], cbam=False)  # [8, 8, 1024]
    center_32 = conv_block(center_32, out_dims_list[4], kernel_size, strides=1)
    center_32 = muti_conv_residual_block(
        center_32, out_dims_list[4], conv_dims_list[4], cbam=False)

    # ------------------------------ 上采样层------------------------------------
    # 上采样层: 原始图像的16倍
    up_16 = layers.Conv2DTranspose(out_dims_list[4], 2, 2, padding='same')(center_32)
    up_16 = muti_conv_residual_block(
        up_16, out_dims_list[4], conv_dims_list[4], cbam=False)
    up_16_c = layers.Concatenate()([out_16_r, up_16])  # [16, 16, 640*4]
    out = conv_block(up_16_c, filters=out_dims_list[4], kernel_size=1)

    # 原始图像的8倍
    up_8 = layers.Conv2DTranspose(out_dims_list[3], 2, 2, padding='same')(out)
    up_8 = muti_conv_residual_block(
        up_8, out_dims_list[3], conv_dims_list[3], cbam=False)
    up_8_c = layers.Concatenate()([up_8, out_8_r])
    out = conv_block(up_8_c, filters=out_dims_list[3], kernel_size=1)

    # 原始图像的4倍
    up_4 = layers.Conv2DTranspose(out_dims_list[2], 2, 2, padding='same')(out)
    up_4 = muti_conv_residual_block(
        up_4, out_dims_list[2], conv_dims_list[2])
    up_4_c = layers.Concatenate()([up_4, out_4_r])  # [64, 64, 384*4]
    out = conv_block(up_4_c, filters=out_dims_list[2], kernel_size=1)

    # 原始图像的2倍
    up_2 = layers.Conv2DTranspose(out_dims_list[1], 2, 2, padding='same')(out)
    up_2 = muti_conv_residual_block(
        up_2, out_dims_list[1], conv_dims_list[1])
    up_2_c = layers.Concatenate()([up_2, out_2_r])  # [64, 64, 384*4]
    out = conv_block(up_2_c, filters=out_dims_list[1], kernel_size=1)

    # 原始图像大小
    up_1 = layers.Conv2DTranspose(out_dims_list[0], 2, 2, padding='same')(out)
    out = muti_conv_residual_block(
        up_1, out_dims_list[0], conv_dims_list[0], cbam=False)

    # 输出层
    out = conv_block(out, num_classes, 1, bn=False, activation='none')
    if num_classes == 1:
        out = layers.Activation('sigmoid')(out)
    else:
        out = layers.Activation('softmax')(out)

    return keras.Model(inputs, out, name='my_model_single_scale')


if __name__ == '__main__':
    my_model_ = my_model((256, 256, 1), dims=32)
    my_model_.summary()

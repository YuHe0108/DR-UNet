import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_package.Module import cbam_block

"""muti-scale  no attention, no aspp"""


# Total params: 12,992,607
# Trainable params: 12,975,327
# Non-trainable params: 17,280

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
        out = attention_block(out, filters)
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

    concat_out = layers.Concatenate()(outputs)  # 特征图拼接
    out = conv_block(concat_out, filters=out_filters, kernel_size=1)  # 整合拼接的特征图
    if cbam:
        out = attention_block(out, out_filters)  # 使用空间和通道注意力集中
    if residual:
        # 是否使用残差结构
        out = layers.Add()([inputs, out])
    return out


def aspp_block(inputs, filters, out_filters, kernel_size):
    # 融合多个尺度的 rate [1, 2, 4, 8]
    out_rate_1 = conv_block(inputs, filters, kernel_size, dilate_rate=1)
    out_rate_2 = conv_block(inputs, filters, kernel_size, dilate_rate=2)
    out_rate_4 = conv_block(inputs, filters, kernel_size, dilate_rate=4)
    out_rate_8 = conv_block(inputs, filters, kernel_size, dilate_rate=8)
    # 特征图拼接
    concat_out = layers.Concatenate()([inputs, out_rate_1, out_rate_2, out_rate_4, out_rate_8])
    # 整合拼接的特征图
    out = conv_block(concat_out, out_filters, kernel_size=1)
    return out


def attention_block(inputs, filters):
    c_out = layers.Multiply()([cbam_block.ChannelAttentionConv(filters)(inputs), inputs])
    out = layers.Multiply()([cbam_block.SpatialAttention()(c_out), c_out])
    return out


def my_model(input_shape, num_classes=1, dims=32, kernel_size=3):
    print(dims)
    conv_dims_list = [32, 64, 96, 128, 160, 192, 224, 320, 416, 512]
    out_dims_list = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280]

    # 输入层
    inputs = layers.Input(input_shape)

    # ------------------------------ 下采样层------------------------------------
    # 初始层: out_2: 下采样2倍
    out_2 = conv_block(inputs, out_dims_list[0], kernel_size, strides=2)
    out_2_r = muti_conv_residual_block(
        out_2, out_dims_list[0], conv_dims_list[0])  # [128, 128, 128]
    out_1_u = layers.UpSampling2D(2, interpolation='bilinear')(out_2_r)

    # out_4: 下采样4倍
    out_4 = conv_block(out_2_r, out_dims_list[1], kernel_size, strides=2)
    out_4_r = muti_conv_residual_block(
        out_4, out_dims_list[1], conv_dims_list[1])  # [64, 64, 256]
    out_2_u = layers.UpSampling2D(2, interpolation='bilinear')(out_4_r)

    # out_8: 下采样8倍
    out_8 = conv_block(out_4_r, out_dims_list[2], kernel_size, strides=2)
    out_8_r = muti_conv_residual_block(
        out_8, out_dims_list[2], conv_dims_list[2])  # [32, 32, 512]
    out_4_u = layers.UpSampling2D(2, interpolation='bilinear')(out_8_r)

    # out_16: 下采样16倍
    out_16 = conv_block(out_8_r, out_dims_list[3], kernel_size, strides=2)
    out_16_r = muti_conv_residual_block(
        out_16, out_dims_list[3], conv_dims_list[3])  # [16, 16, 768]
    out_8_u = layers.UpSampling2D(2, interpolation='bilinear')(out_16_r)

    # out_32: 下采样32倍
    out_32 = conv_block(out_16_r, out_dims_list[4], kernel_size, strides=2)
    out_32_r = muti_conv_residual_block(
        out_32, out_dims_list[4], conv_dims_list[4])  # [8, 8, 1024]
    out_16_u = layers.UpSampling2D(2, interpolation='bilinear')(out_32_r)

    # ------------------------------ 中间层------------------------------------
    # 融合多个尺度的 rate [1, 2, 4, 8]
    out = aspp_block(out_32_r, conv_dims_list[4], out_dims_list[4], kernel_size)
    # 使用空间和通道注意力集中
    out = attention_block(out, out_dims_list[4])
    # 残差连接
    dilate_out_1 = layers.Add()([out_32_r, out])
    rate_1_u = layers.UpSampling2D(2, interpolation='bilinear')(dilate_out_1)

    # 第二个中间层
    out = aspp_block(dilate_out_1, conv_dims_list[4], out_dims_list[4], kernel_size)
    # 使用空间和通道注意力集中
    out = attention_block(out, out_dims_list[4])
    # 残差连接
    out = layers.Add()([dilate_out_1, out])
    rate_2_u = layers.UpSampling2D(2, interpolation='bilinear')(out)

    # ------------------------------ 上采样层------------------------------------
    # 上采样层: 原始图像的16倍
    up_16 = layers.Conv2DTranspose(out_dims_list[4], 2, 2, padding='same')(out)
    up_16 = muti_conv_residual_block(
        up_16, out_dims_list[4], conv_dims_list[4])
    up_16_c = layers.Concatenate()([up_16, out_16_r, out_16_u, rate_1_u, rate_2_u])  # [16, 16, 640*4]
    # 特征聚合
    out = conv_block(up_16_c, filters=out_dims_list[4], kernel_size=1)
    # 使用空间和通道注意力集中、残差连接
    out = attention_block(out, out_dims_list[4])
    out = layers.Add()([up_16, out])

    # 原始图像的8倍
    up_8 = layers.Conv2DTranspose(out_dims_list[3], 2, 2, padding='same')(out)
    up_8 = muti_conv_residual_block(
        up_8, out_dims_list[3], conv_dims_list[3])
    up_8_c = layers.Concatenate()([up_8, out_8_r, out_8_u])  # [32, 32, 512*4]
    # 特征聚合
    out = conv_block(up_8_c, filters=out_dims_list[3], kernel_size=1)
    # 使用空间和通道注意力集中
    out = attention_block(out, out_dims_list[3])
    # 残差连接
    out = layers.Add()([up_8, out])

    # 原始图像的4倍
    up_4 = layers.Conv2DTranspose(out_dims_list[2], 2, 2, padding='same')(out)
    up_4 = muti_conv_residual_block(
        up_4, out_dims_list[2], conv_dims_list[2])
    # 特征拼接、聚合
    up_4_c = layers.Concatenate()([up_4, out_4_r, out_4_u])  # [64, 64, 384*4]
    out = conv_block(up_4_c, filters=out_dims_list[2], kernel_size=1)
    # 使用空间和通道注意力集中、残差结构
    out = attention_block(out, out_dims_list[2])
    out = layers.Add()([up_4, out])

    # 原始图像的2倍
    up_2 = layers.Conv2DTranspose(out_dims_list[1], 2, 2, padding='same')(out)
    up_2 = muti_conv_residual_block(
        up_2, out_dims_list[1], conv_dims_list[1])
    up_2_c = layers.Concatenate()([up_2, out_2_r, out_2_u])  # [64, 64, 384*4]
    out = conv_block(up_2_c, filters=out_dims_list[1], kernel_size=1)
    # 使用空间和通道注意力集中
    out = attention_block(out, out_dims_list[1])
    out = layers.Add()([up_2, out])

    # 原始图像大小
    up_1 = layers.Conv2DTranspose(out_dims_list[0], 2, 2, padding='same')(out)
    up_1 = muti_conv_residual_block(up_1, out_dims_list[0], conv_dims_list[0])
    # 特征拼接
    up_1_c = layers.Concatenate()([up_1, out_1_u])  # [64, 64, 384*4]
    # 特征聚合
    out = conv_block(up_1_c, filters=out_dims_list[0], kernel_size=1)
    # 使用空间和通道注意力集中
    out = attention_block(out, out_dims_list[0])
    # 残差连接
    out = layers.Add()([up_1, out])

    # 输出层
    out = conv_block(out, num_classes, 1, bn=False, activation='none')
    if num_classes == 1:
        out = layers.Activation('sigmoid')(out)
    else:
        out = layers.Activation('softmax')(out)

    return keras.Model(inputs, out, name='my_model_all')


if __name__ == '__main__':
    my_model_ = my_model((256, 256, 1), dims=32)
    my_model_.summary()

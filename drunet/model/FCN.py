import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def get_encoder(input_shape, pretrained='imagenet'):
    """将vgg16作为图像特征提取器， 输出五种不同尺度的特征图"""
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0
    assert input_shape[2] == 3
    weight_path = r'C:\Users\YingYing\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    img_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x  # [b, 32, 32, 256]

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x  # [b, 16, 16, 512]

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    f5 = x  # [b, 8, 8, 512]

    if pretrained == 'imagenet':
        keras.Model(img_input, x).load_weights(weight_path)
    return img_input, [f1, f2, f3, f4, f5]


def crop(o1, o2, i):
    """裁剪o1和o2至相同的尺寸，裁剪的宽和高取两个特征图的最小值"""
    o_shape2 = keras.Model(i, o2).output_shape
    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = keras.Model(i, o1).output_shape
    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = layers.Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = layers.Cropping2D(cropping=((0, 0), (0, cx)))(o2)
    if output_height1 > output_height2:
        o1 = layers.Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = layers.Cropping2D(cropping=((0, cy), (0, 0)))(o2)
    return o1, o2


def fcn_8_vgg(input_shape, num_classes, pretrained='imagenet'):
    img_input, levels = get_encoder(input_shape, pretrained=pretrained)
    [f1, f2, f3, f4, f5] = levels
    o = f5

    o = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)
    o = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = layers.Dropout(0.5)(o)

    # 上采样
    o = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o)
    o = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f4
    o2 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o2)
    o, o2 = crop(o, o2, img_input)
    o = layers.Add()([o, o2])

    # 上采样
    o = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f3
    o2 = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(o2)
    o2, o = crop(o2, o, img_input)
    o = layers.Add()([o2, o])

    # 上采样
    o = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8),
                               use_bias=False, padding='same')(o)
    if num_classes == 1:
        o = layers.Activation('sigmoid')(o)
    else:
        o = layers.Softmax()(o)

    model = keras.Model(img_input, o, name='fcn8s')
    return model


if __name__ == '__main__':
    fcn_8_model = fcn_8_vgg((256, 256, 3), 1)
    fcn_8_model.summary()
    # m = fcn_32(101)

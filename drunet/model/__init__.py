from tf_package.Segment import models

from . import dr_unet
from . import normal as my_model


def get_seg_model(model_name, input_shape, dims=64, num_classes=1, **kwargs):
    if model_name == 'my_model':
        seg_model = my_model.my_model(input_shape, num_classes, dims)
    elif model_name == 'dr_unet':
        seg_model = dr_unet.my_model(input_shape)
    else:
        seg_model = models.get_seg_model(model_name=model_name,
                                         input_shape=input_shape,
                                         num_classes=num_classes,
                                         dims=dims, **kwargs)
    return seg_model

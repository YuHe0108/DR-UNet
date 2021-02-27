# DR-UNet: A robust deep learning segmentation method for hematoma volumetric detection in intracerebral hemorrhage

------

## Structure of DR-UNet

This is an implementation of  DR-UNet on Python 3.6, Keras, and TensorFlow2.0. DR-UNet consists of an encoding (down-sampling) path and a decoding (up-sampling) path. The model structure is shown in the figure below.



![model structure](figures/Fig0.jpg)

To increase the segmentation performance of the model, three reduced dimensional residual convolution units (RDRCUs) were developed to replace the traditional convolution layer. The three convolution blocks are illustrated in the following figure. The three RDRCUs have two branches (main branch and side branch) to process the input characteristics continuously.

![model structure](figures/Fig7.jpg)



## Getting Started

<img src="figures/Fig1.jpg" alt="model structure " style="zoom:50%;" />


<img src="figures/Fig2.jpg" alt="model structure" style="zoom:50%;" />


![model structure](figures/Fig3.jpg)


![model structure](figures/Fig4.jpg)


![model structure](figures/Fig5.jpg)


![model structure](figures/Fig6.png)













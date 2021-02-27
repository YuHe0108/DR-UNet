# DR-UNet: A robust deep learning segmentation method for hematoma volumetric detection in intracerebral hemorrhage

------

## Structure of DR-UNet

This is an implementation of  DR-UNet on Python 3.6, Keras, and TensorFlow2.0. DR-UNet consists of an encoding (down-sampling) path and a decoding (up-sampling) path. The model structure is shown in the figure below.



![model structure](figures/Fig0.jpg)

To increase the segmentation performance of the model, three reduced dimensional residual convolution units (RDRCUs) were developed to replace the traditional convolution layer. The three convolution blocks are illustrated in the following figure. The three RDRCUs have two branches (main branch and side branch) to process the input characteristics continuously.

![model structure](figures/Fig7.jpg)

## Experimental results of hematoma segmentation

We first trained DR-UNet to recognize the hematoma region in patients. The performance was evaluated on two testing datasets (internal and external) using the following criteria: i) sensitivity, ii) specificity, iii) precision, iv) Dice, v) Jaccard and vi) VOE (details in the Methods section). Moreover, we compared DR-UNet with UNet, FCM and active contours. In all four methods, segmentation labeling was considered the ground truth standard (details in the Methods section). The main calculation results are shown in the figure and table below.

As shown in the table below, results of sensitivity, specificity, precision, Dice, Jaccard and VOE by four methods in the internal testing and the external testing dataset.

![model structure](figures/Fig8.jpg)

Figure A shows the boxplots for the performance of the DR-UNet models and the other three methods for the segmentation and detection of ICHs on the two testing datasets. The internal testing dataset in the retrospective dataset was enriched to include all ICH subtypes. In Figure B, four different types of hematomas were included, and we visually presented a performance comparison among the DR-UNet, UNet, FCM and active contour methods.

<img src="figures/Fig1.jpg" alt="model structure " style="zoom:90%;" />



<img src="figures/Fig2.jpg" alt="model structure" style="zoom:50%;" />


![model structure](figures/Fig3.jpg)


![model structure](figures/Fig4.jpg)


![model structure](figures/Fig5.jpg)


![model structure](figures/Fig6.png)













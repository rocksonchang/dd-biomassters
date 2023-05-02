# Driven Data - BioMassters Competition

## Challenge

> The goal of the BioMassters challenge was to estimate the yearly biomass of 2,560 meter by 2,560 meter patches of land in Finland's forests using Sentinel-1 and Sentinel-2 imagery of the same areas on a monthly basis. The ground truth for this competition came from LiDAR (Light Detection And Ranging) and in-situ measurements of the same forests collected by the Finnish Forest Centre on an annual basis.

[Competition Page](https://www.drivendata.org/competitions/99/biomass-estimation/)

Feature data consists of monthly above ground biomass imagery (AGBM) from Sentinel-1 and Sentinel-2 (4 and 11 channels, respectively), taken over the span of 12 months (24 images total). Each image is 256 by 256 pixels.

Target data consists of ground-truth AGBM measurements collected using LiDAR in the form of a 256 by 256 pixel image.

## Approach

The approach taken was to treat this as an image translation problem: from multi-dimensional feature data space (12 months x 15 channels) to single-dimension target image. Two approaches were considered: 

1. UNET encoder-decoder network
   
   A simple encoder-decoder network with skip connections that allow for the effficient passage of lower-level image features from the encoder into the decoder network. This is particularly useful in this case as the feature and target data share a lot of features (i.e. geographic).

2. [Pix2Pix](https://arxiv.org/abs/1611.07004), a conditional generative adverserial network (cGAN)

    A more complex network consisting of an encoder-decoder and an adverserially trained discriminator. In principal, the combined training should result in higher quality images. However these networks can be tricky to train owing to the  complexity of the loss function. 

In experimentation, the added compleity of the cGAN did not significantly improve validation results and so most of the focus was placed on the encoder-decoder network.

## Implementation

The impelmentation is based on the [Pix2Pix pytorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This provides a lot of supporting code for model eval and training, while providing a framework to customize data loading and custom models.

**Modifications Over Source**
- `BioMasstersDownloader` to support the downloading of data to local storage. This [notebook](https://github.com/rocksonchang/dd-biomassters/blob/master/pix2pix/download-biomassters-data.ipynb) provides examples of how to work with this module.
- `BioMasstersDataset` class to support loading of data into memory. This is implemented as a Pytorch `Dataset`.
- `Pix2PixBioModel` model class for the cGAN model adapted to the BioMassters challenge. Additionally, support for metric visualization and loss function exploration was added over the original model class.
- `EncodeDecodeBioModel` model class for the UNET encoder-decoder model adapted to the BioMassters challenge. Additionally, support for metric visualization and loss function exploration was added over the original model class.

## Training
Models were trained using AWS `Deep Learning AMI GPU PyTorch 1.13.0` images on EC2 `g4dn.xlarge` instances. 

See [notebook](https://github.com/rocksonchang/dd-biomassters/blob/master/notebooks/run-train-test.ipynb) for examples on how to run.

Note: Run `python -m visdom.server` to start [visdom](https://github.com/fossasia/visdom) server and open `localhost:8097` to monitor training results.

## Results
The best performing model scored 34 average RMSE, a fair score but short of the [top contributions](https://github.com/drivendataorg/the-biomassters/tree/main).

The under current implementation and hardware, the cGAN model did not show significant improvement over the simpler UNET model.

A major bottleneck in the above approach was handling the high-dimensional input (180 channels from 12 images and 15 image channels). A larger and deeper network may have been better able to handle this size input, but would have also required more training time and more powerful hardware. 

**Insights from the [winning contribution](https://github.com/drivendataorg/the-biomassters/tree/main/1st-place)**:
- UNET architecture
- used a self-attention mechanism over the monthly images at the encoder.
- significant data augmentation during training
- mixed precision training
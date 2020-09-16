# PiePline toolbelt

# Installation:
[![PyPI version](https://badge.fury.io/py/pietoolbelt.svg)](https://badge.fury.io/py/pietoolbelt)
[![PyPI Downloads/Month](https://pepy.tech/badge/pietoolbelt/month)](https://pepy.tech/project/pietoolbelt)
[![PyPI Downloads](https://pepy.tech/badge/pietoolbelt)](https://pepy.tech/project/pietoolbelt)

`pip install pietoolbelt`

##### Install latest version before it's published on PyPi
`pip install -U git+https://github.com/PiePline/pietoolbelt`

# Functional
* `augmentations`
    * `augmentations.detection` - predefined augmentations for detection task
    * `augmentations.segmentation` - predefined augmentations for segmentation task
* `datasets`
    * `datasets.stratification` - stratification by histogram
    * `datasets.utils` - set of datasets constructors that
* `losses`
    * `losses.common` - losses utils
    * `losses.regression` - regression losses
    * `losses.segmentation` - losses for single and multi-class segmentation
    * `losses.detection` - losses for detection task
* `metrics` - 
    * `metrics.common` - common utils for metrics
    * `cpu` - metrics, that calculates by `numpy`
        * `metrics.cpu.classification` - classification metrics
        * `metrics.cpu.detection` - detection metrics
        * `metrics.cpu.regression` - regression metrics
        * `metrics.cpu.segmentation` - segmentation metrics
    * `torch` - metrics, that calculates by `torch`
        * `metrics.torch.classification` - classification metrics
        * `metrics.torch.detection` - detection metrics
        * `metrics.torch.regression` - regression metrics
        * `metrics.torch.segmentation` - segmentation metrics
* `models` - models zoo and constructors
    * `decoders.unet` - UNet decoder, that automatically constructs by encoder
    * `encoders.common` - basic interfaces for encoders
    * `encoders.inception` - InceptionV3 encoder
    * `encoders.mobile_net` - MobileNetV2 encoder
    * `encoders.resnet` - ResNet encoders
    * `albunet` - albunet model
    * `utils` - models utils
    * `weights_storage` - pretrained weights storage
* `steps` - some training process steps
    * `regression.train` - train step for regression task
    * `regression.bagging` - bagging step for regression task
    * `segmentation.bagging` - bagging step for segmentation task
    * `segmentation.inference` - inference for segmentation model
    * `segmentation.predict` - predict step for segmentation task
    * `stratification` - dataset stratification class
* `img_matcher` - image comparison and matching tool based on descriptors
* `mask_composer` - mask composer tools that can effectively combine masks for regular, instance or multiclass segmentation
* `train_config` - some predefined train configs for [PiePline](https://github.com/PiePline/piepline)
* `tta` - test time augmentations 
* `utils` - some utils
* `viz` - image visualisation tools

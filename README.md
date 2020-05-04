# PiePline toolbelt

# Installation:
[![PyPI version](https://badge.fury.io/py/pietoolbelt.svg)](https://badge.fury.io/py/pietoolbelt)
[![PyPI Downloads/Month](https://pepy.tech/badge/pietoolbelt/month)](https://pepy.tech/project/pietoolbelt)
[![PyPI Downloads](https://pepy.tech/badge/pietoolbelt)](https://pepy.tech/project/pietoolbelt)

`pip install pietoolbelt`

##### Install latest version before it's published on PyPi
`pip install -U git+https://github.com/PiePline/pietoolbelt`

# Functional
* Datasets
    * `datasets.stratification` - stratification by histogram
    * `datasets.utils` - set of datasets constructors that
* Losses
    * `losses.common` - losses utils
    * `losses.regression` - regression losses
    * `losses.segmentation` - losses for single and multi-class segmentation
    * `losses.detection` - losses for detection task
* Metrics
    * `metrics.common` - common utils for metrics
    * CPU - metrics, that calculates by `numpy`
        * `metrics.cpu.classification` - classification metrics
        * `metrics.cpu.detection` - detection metrics
        * `metrics.cpu.regression` - regression metrics
        * `metrics.cpu.segmentation` - segmentation metrics
    * Torch - metrics, that calculates by `torch`
        * `metrics.torch.classification` - classification metrics
        * `metrics.torch.detection` - detection metrics
        * `metrics.torch.regression` - regression metrics
        * `metrics.torch.segmentation` - segmentation metrics
* Models
    * `decoders.unet` - UNet decoder, that automatically constructs by encoder
    * `encoders.common` - basic interfaces for encoders
    * `encoders.inception` - Inceptionv3 encoder
    * `encoders.mobile_net` - MobileNetv2 encoder
    * `encoders.resnet` - ResNet encoders
    * `albunet` - albunet model
    * `utils` - models utils
    * `weights_storage` - pretrained weights storage
* Pipeline steps
    * `regression.train` - train step for regression task
    * `regression.bagging` - bagging step for regression task
* `img_matcher` - image comparision and matching tool based on descriptors
* `mask_composer` - mask composer tools that can effectively combine masks for regular, instance or multiclass segmentation
* `utils` - some utils
* `viz` - image visualisation tools

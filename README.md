## This is the source code for muliple-view classifiers

It supports all classifiers strategies from the research papers below.
The code is an evolution of the code used in these papers.

## This code supports:
 - Patch classifier
 - Single image classifier (using patches com patch-classifier or pretrained from ImageNet)
 - Two views classifier (using model from single-image-classifier above)

## This code also supports many different strategies:
 - It can perform all classifiers above with different backbones, including timm flavors. You can run using tenths of backbones. Although transformer-based ones is supported only for patch-based classifiers (you are invited to change this)
 - Supports image resize to double size or half size, using two approaches fixed-resize and Learn-to-resize (comparing with default 1.152x896)
 - Supports multiple datasets (CBIS-DDSM and VinDr-Mammo in current version)


### Reference
If you use want to know more, please check complete text [HERE](https://arxiv.org/abs/2503.19945). If you want to cite this work please use reference below.

```
@misc{petrini2025optimizingbreastcancerdetection,
      title={Optimizing Breast Cancer Detection in Mammograms: A Comprehensive Study of Transfer Learning, Resolution Reduction, and Multi-View Classification}, 
      author={Daniel G. P. Petrini and Hae Yong Kim},
      year={2025},
      eprint={2503.19945},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2503.19945}, 
}
```


And [here](https://ieeexplore.ieee.org/document/9837037). If you want to cite this work please use reference below.

```
@ARTICLE{
9837037,
  author={Petrini, Daniel G. P. and Shimizu, Carlos and Roela, Rosimeire A. and Valente, Gabriel Vansuita and Folgueira, Maria Aparecida Azevedo Koike and Kim, Hae Yong},
  journal={IEEE Access}, 
  title={Breast Cancer Diagnosis in Two-View Mammography Using End-to-End Trained EfficientNet-Based Convolutional Network}, 
  year={2022},
  volume={10},
  number={},
  pages={77723-77731},
  keywords={Mammography;Convolutional neural networks;Training;Transfer learning;Breast cancer;Artificial intelligence;Lesions;Breast cancer diagnosis;deep learning;convolutional neural network;mammogram;transfer learning},
  doi={10.1109/ACCESS.2022.3193250}
}


```
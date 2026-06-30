## This is the source code for muliple-view classifiers

It supports all classifiers strategies from the research papers below.
The code is an evolution of the code used in these papers.

## This code supports:
-Patch classifier
-Single image classifier (using patches com patch-classifier or pretrained from ImageNet)
-Two views classifier (using model from single-image-classifier above)

## This code also supports many different strategies:
-It can perform all classifiers above with different backbones, including timm flavors. You can run using tenths of backbones. Although transformer-based ones is supported only for patch-based classifiers (you are invited to change this)
-Supports image resize to double size or half size, using two approaches fixed-resize and Learn-to-resize (comparing with default 1.152x896)
-Supports multiple datasets (CBIS-DDSM and VinDr-Mammo in current version)
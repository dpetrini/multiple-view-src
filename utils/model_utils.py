
# Versão otimizada mar/2024 - usando apenas models default PytTorch
#                           (não usa timm nem Efficientnet local para rede inteira, sim para bloco)
#

import copy
import torch
import torch.nn as nn

import torchvision.models as models

# from efficientnet_pytorch import EfficientNet

# https://github.com/rwightman/pytorch-image-models
from timm import create_model

from resnet_blocks import ResnetBlocks
from efficientnet_blocks import EFBlocks

TOP_MODE = 'TOP_EF_NET'   # 'TOP_RESNET' 'TOP_EF_NET'
TOP_LAYER_N_BLOCKS = 2
STRIDES = 2

def load_model(mode, network, model_file, weights=True,
               patch_weights=False,         # can load model from patch or from different single model
               top_mode=TOP_MODE, top_layer_blocks=TOP_LAYER_N_BLOCKS,
               strides=STRIDES, num_classes=5):
    """
    Load models for breast classifiers.
    Only operate on cpu, you should load after this function. (since 03-2026)
    Supports timm models directly, at least it should
    """
    extract_layers = 2

    print(f'Loading [{mode}] model: {network} file: {model_file} for Single image clf')

    if 'Resnet' in network:
        if network == 'Resnet18':
            # model = models.resnet18(pretrained=weights)
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)
            extract_layers = 2
            inplanes = 512
        if network == 'Resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if weights else None)
            extract_layers = 2
            inplanes = 2048
        if network == 'Resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if weights else None)
            extract_layers = 2
            inplanes = 2048
        num_ftrs = model.fc.in_features     # 2048
        model.fc = nn.Linear(num_ftrs, num_classes)   # change for our 5 categories

    elif network == 'mobilenet_v2':
        # model = models.mobilenet_v2(pretrained=weights)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if weights else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if weights else None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 2

    elif network == 'alexnet':
        model = models.alexnet(pretrained=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 2

    elif network == 'mnasnet1_0':
        model = models.mnasnet1_0(pretrained=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'inception_v3': # Does not work
        model = models.inception_v3(pretrained=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 2

    elif network == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if weights else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'densenet169':
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if weights else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if weights else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        inplanes = num_ftrs
        extract_layers = 1

    elif network == 'ResNest50':
        # !pip install git+https://github.com/facebookresearch/fvcore.git
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=weights)
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=weights)
        num_ftrs = model.fc.in_features     # 2048
        model.fc = nn.Linear(num_ftrs, num_classes)   # change for our 5 categories
        inplanes = num_ftrs
        extract_layers = 2

    # Not separating behavior for efficient net, using pytorch models
    # See https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py for imagenet accuracies & input sizes
    elif 'EfficientNet' in network:
        if network == 'EfficientNet_b7':
            model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT if weights else None)
        if network == 'EfficientNet_b6':
            model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT if weights else None)
        elif network == 'EfficientNet_b5':
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if weights else None)
        elif network == 'EfficientNet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if weights else None)
        elif network == 'EfficientNet_b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if weights else None)
        elif network == 'EfficientNet_b2':
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if weights else None)
        elif network == 'EfficientNet_b1':
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if weights else None) # only B1...V2
        elif network == 'EfficientNet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if weights else None)
        # Create MLP head and save for assemble Single Clf
        inplanes = model.classifier[1].in_features
        extract_layers = 2
        model.classifier[1] = nn.Linear(inplanes, num_classes)

    elif 'efficientnet_v2' in network:
        if network == 'efficientnet_v2_s':   # size: 384x384
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if weights else None)
            inplanes = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(inplanes, num_classes)
        elif network == 'efficientnet_v2_m':  # size: 480x480
            model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if weights else None)
            inplanes = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(inplanes, num_classes)
        elif network == 'tf_efficientnetv2_s.in21kperfeitos.nvidia l40 ':
            model = create_model('tf_efficientnetv2_s.in21k', pretrained=True)
            inplanes = model.classifier.in_features
            model.classifier = nn.Linear(inplanes, num_classes, bias=True)  
        extract_layers = 2

    elif 'convnext' in network:
        if network == 'convnext_tiny':
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if weights else None)
            inplanes = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(inplanes, num_classes)
        elif network == 'convnext_small':
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if weights else None)
            inplanes = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(inplanes, num_classes)
        elif network == 'convnext_base':
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if weights else None)
            inplanes = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(inplanes, num_classes)
        elif network == 'convnext_tiny_in22k':
            model = create_model("convnext_tiny_in22k", pretrained=True)
            inplanes = model.head.fc.in_features
            model.head.fc = nn.Linear(inplanes, num_classes, bias=True)
        elif network == 'convnext_small_in22k':
            model = create_model("convnext_small_in22k", pretrained=True)
            inplanes = model.head.fc.in_features
            model.head.fc = nn.Linear(inplanes, num_classes, bias=True)            
        elif network == 'convnext_base_in22k':
            model = create_model("convnext_base_in22k", pretrained=True)
            inplanes = model.head.fc.in_features
            model.head.fc = nn.Linear(inplanes, num_classes, bias=True)            
        extract_layers = 2

    elif 'swin-v2' in network:
        if network == 'swin-v2_tiny_1k':
            model = create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
        elif network == 'swin-v2_base_in22k_ft_1k':
            model = create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
        inplanes = model.head.fc.in_features
        model.head.fc = nn.Linear(inplanes, num_classes, bias=True)
        extract_layers = 2  

    elif 'vit-base-clip' in network:
        if network == 'vit-base-clip-laion-16':
            model = create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        elif network == 'vit-base-clip-laion-32':
            model = create_model('vit_base_patch32_clip_224.laion2b', pretrained=True)
        inplanes = model.head.in_features
        model.head = nn.Linear(inplanes, num_classes, bias=True)
        extract_layers = 2  

    else:
        try:                                        # Inserted in 2025-11-20
            # Try create timm model anyway
            print('Trying model from Timm: ', network)
            model = create_model(network, pretrained=True, num_classes=num_classes)
            inplanes = model.num_features       # Usually ok for timm
            if 'mobilenetv4_conv_small' in network:     # Special case, if more cases place here
                extract_layers = 6   # removes classifier,flatten,act2,norm_head,conv_head,global_pool
                #print('mobile...', extract_layers, inplanes)
            else:
                # get number of linear layers to discard
                model_reset = copy.deepcopy(model)
                model_reset.reset_classifier(0)     # take only features
                before_keys = set(model.state_dict().keys())
                after_keys = set(model_reset.state_dict().keys())
                extract_layers = len(before_keys - after_keys)
                print("Linear layers removed to asssemble full clf: ", extract_layers)
                del model_reset     # free memory
        except Exception:
            try:
                # Try create timm model anyway
                print('Trying model from PyTorch models: ', network)
                model = getattr(models, network)(weights=True)
            except :
                raise ValueError(f"Model '{network}' not found anyway.")
 
    # until here loaded vanilla model with or without default pretrain  #####
 
    if mode == 'patch':
        return model

    if mode == 'single_pure':
        return model, inplanes, extract_layers

    if mode == 'transfer':
        # Load Full single model after network assembled
        print('\nLoading model for transfer learning.\n')
        # model.load_state_dict(torch.load(model_file, map_location=device))  31-03-2026
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print(inplanes, extract_layers)
        return model, inplanes, extract_layers

    if mode == 'learn_resize_input':
        from learn_resize import ResizingNetwork
        print('Using Learn to Resize model (0.5)')

        class LearnResizeFull(nn.Module):
            def __init__(self,):
                super(LearnResizeFull, self).__init__()
                self.resizer = ResizingNetwork(r=2, n=16, in_channels=1)
                self.model = model
            def forward(self, x):
                x = self.resizer(x)
                x = self.model(x)
                return x

        learned_model = LearnResizeFull()
        return learned_model

    # Fixed Resizer April-2024
    if mode == 'fixed_resize_input':
        from learn_resize import FixedResizeNetwork
        print('Using Fixed Resize model (0.5)')

        class FixedResizeFull(nn.Module):
            def __init__(self,):
                super(FixedResizeFull, self).__init__()
                self.resizer = FixedResizeNetwork()
                self.model = model
            def forward(self, x):
                x = self.resizer(x)
                x = self.model(x)
                return x

        fixed_resize_model = FixedResizeFull()
        return fixed_resize_model

    if mode == 'single':  # Full img classifier for Patch based

        print(f'Loading [{mode}] model: {network} patch model file: {model_file} for Patch based classifier') #2026
        print(f' Using inplanes: {inplanes} and extract_layers: {extract_layers}.')

        if patch_weights:
            # Load patch  model from file for training single
            # model.load_state_dict(torch.load(model_file, map_location=device))  #31-03-2026
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
        # Define Top Layer
        if top_mode == 'TOP_RESNET':
            top_layer = ResnetBlocks(inplanes=inplanes)
        elif top_mode == 'TOP_EF_NET':
            print('Using MBConv TopLayer')
            top_layer = EFBlocks([15, 15], inplanes=inplanes, outplanes=inplanes, 
                                 n_blocks=top_layer_blocks, strides=strides)

        class FullClassifier(nn.Module):
            def __init__(self, net):
                super(FullClassifier, self).__init__()
                self.net = net
                print(self.net)
                # Essa linha copia as camadas com os pesos já carregados.
                self.feature_extractor = nn.Sequential(*list(model.children())[:-extract_layers])
                self.top_layer = top_layer

            def forward(self, x):
                x = self.feature_extractor(x)
                x = self.top_layer(x)
                return x

        model = FullClassifier(network)   # Essa linha não está carregando um modelo nao inicializado?

        if not patch_weights:
            # Load Full single model after network assembled (single for patch based - two views training)
            # model.load_state_dict(torch.load(model_file, map_location=device)) # 31-03-2026
            model.load_state_dict(torch.load(model_file, map_location='cpu'))

        return model, inplanes, extract_layers

    return model

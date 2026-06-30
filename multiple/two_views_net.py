# Side Classifier: 2 views classifier using Sinai Patch Clf for CC, MLO

import sys
import collections
import torch
import torch.nn as nn

from bottleneck_ import Bottleneck
from efficientnet_pytorch import EfficientNet, MBConvBlock
from utils.model_utils import load_model

class TwoViewsMIDBreastClassifier(nn.Module):
    """
    Two-views:
    1-Take Feature Extractor part from Full image Classifier (which is the patch classifier part).
    2-Concatenate the outputs (activation maps) and output.
    Topology = MID. Means we concatenate feature extractor from each side and later will stack 
    top layers in the "middle of the way"
    """
    def __init__(self, device, model_file, network, exp_type = 'CVUBP', dataset='CBIS-DDSM'):
                #  _extract_layers = 2):
        super(TwoViewsMIDBreastClassifier, self).__init__()
        if exp_type == 'SINGLE_PURE':
            self.single_clf_full, _, extract_layers = load_model('transfer', network, 
                                                                  model_file, weights=True,
                                                                  num_classes=2)
            self.feature_extractor = nn.Sequential(*list(self.single_clf_full.children())[:-extract_layers])
            self.single_clf_core = self.feature_extractor
        elif exp_type == 'PATCH_BASED':
                # Reconstruct FullClassifier architecture and load checkpoint.
                # 'transfer' mode would create a vanilla backbone (keys: features.*)
                # but the saved checkpoint is a FullClassifier (keys: feature_extractor.*).
                # 'single' mode with patch_weights=False builds FullClassifier then loads model_file.
                self.single_clf_full, _, _ = load_model('single', network, model_file,
                                                         weights=False, patch_weights=False,
                                                         num_classes=2)
                self.single_clf_core = self.single_clf_full.feature_extractor
        else:
            # Experiments before 2024
            self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))  
            self.single_clf_core = self.single_clf_full.feature_extractor  
    def forward(self, x):
        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        hidden = torch.cat([x1_2, x2_2], dim=1)
        return hidden


class SideMIDBreastModel(nn.Module):
    """ Calls TwoViewsMIDBreastClassifier that instantiates 2-views (CC+MLO) with
        concatenated output (without their original top layer).
        Then append the 2-views top layer, that can be resblocks or MBConv blocks,
        with 1, 2 or 0 count. The last means only FC layer.
        Strides: no. os strides for EficientNets
    """
    connections = 256
    output_size = (4, 2)
    def __init__(self, device, model_file, network, n_blocks, b_type='resnet', 
                 avg_pool=True, strides=1, exp_type = 'PATCH_BASED',
                  dataset='CBIS-DDSM', inplanes = 2048, extract_layers = 2):
        super(SideMIDBreastModel, self).__init__()
        if n_blocks not in [0, 1, 2]:
            print('Wrong number of Top Layer blocks.')
            sys.exit()
        self.n_blocks = n_blocks

        # Get two input legs from main classifier:
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network,
                                                         exp_type, dataset)
        self.avg_pool = avg_pool
        output_channels = 2048
        print('Creating Side Mid Networking using:', network, ' and Top Block type: ', b_type, ' Qty: ', n_blocks)
        if network == 'Resnet50':
            input_channels = 4096       # From concatenation
            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output  
           
            if n_blocks == 1:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
            elif n_blocks == 2:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)
        else:
            # Other networks using Resnet Blocks
            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output
                input_channels = 3584       # From concatenation+
                if n_blocks == 1:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                elif n_blocks == 2:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                    self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)
                else:
                    output_channels = 3584  # only FC
            # Other networks using mbconv Blocks
            elif b_type == 'mbconv':
                print('Using EFBlocks (top block) parameters from EfficientNet-b0 [Two views creation].')
                feat_ext = EfficientNet.from_name('efficientnet-b0', num_classes=5) 
                _inplanes = 2 * inplanes
                output_channels = _inplanes
                self.w_h = 36*28                 # width and height of last layer output
                # Parameters for an individual model block
                BlockArgs = collections.namedtuple('BlockArgs', [
                    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])
                block_args = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides],
                                      expand_ratio=2, input_filters=output_channels, #inplanes, CLEAN
                                      output_filters=output_channels, se_ratio=0.25,
                                      id_skip=True) # for 1 block
                # below line for same params for blocks as main net
                global_params, image_size = feat_ext._global_params, [15, 15]
                if n_blocks == 1:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size)
                elif n_blocks == 2:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size)
                    self.block2 = MBConvBlock(block_args, global_params, image_size=image_size)
        if self.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(output_channels, 2)
        else:
            # AVGPOOL
            self.fc_pre = nn.Linear(2048* self.w_h, 1024)  #para aproveitar features espaciais Resnet 9*7 / EficientNet 36*28
            self.fc = nn.Linear(1024, 2)     # INCLUINDO - 2020-10-14 - para aproveitar features espaciais

    def forward(self, x):
        # x.shape: torch.Size([1, 3, 1152, 896])
        x = self.two_views_clf(x)       # out: torch.Size([1, 4096, 36, 28])
        if self.n_blocks == 1:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
        elif self.n_blocks == 2:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
            x = self.block2(x)          # out: Resnet([1, 2048, 9, 7]) / Eficientnet [2, 2048, 36, 28] 
        if self.avg_pool:
            x = self.avgpool(x)             # out: torch.Size([1, 2048, 1, 1])
            x = torch.flatten(x, 1)         # out: torch.Size([1, 2048])
        else:
            # NO AVGPOOL
            x = x.view(-1, 2048* self.w_h)  # para aproveitar features espaciais  Resnet 9* 7 / Efiencet 36*28
            x = self.fc_pre(x)
        x = self.fc(x)                  # out: torch.Size([1, 2]

        return x


# Below not supported now

class SideTOPBreastModel(nn.Module):
    # Only remove FCs from single classifiers - not supported now
    connections = 256
    output_size = (4, 2)
    def __init__(self, device, model_file, network):
        super(SideTOPBreastModel, self).__init__()
        self.two_views_clf = TwoViewsTOPBreastClassifier(device, model_file, network)
        self.fc = nn.Linear(2048*2, 2)
    def forward(self, x):
        x = self.two_views_clf(x)
        #x_side = torch.cat([x[SIDES.CC], x[SIDES.MLO]], dim=1)
        # x_side.shape: torch.Size([1, 4096, 1, 1])
        # x = torch.flatten(x_side, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x.shape: torch.Size([1, 2])
        return x

# IMPLEMENTA APENAS FC depois dos blocos. Com e sem AVGPool.
# Ok para eficient net
class SideFCBreastModel(nn.Module):
    def __init__(self, device, model_file, network, avgpool=False):
        super(SideFCBreastModel, self).__init__()
        self.avgpool_flag = avgpool
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network)
        if self.avgpool_flag:
            if 'b0' in network:
                inplanes = 1280*2     # tamanho dos mapas concatenados do MIDBreast avg 2D
            elif 'b4' in network:
                inplanes = 1280*2 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if 'b0' in network:
                inplanes = 1280*2*36*28     # tamanho dos mapas concatenados do MIDBreast
            elif 'b4' in network:
                inplanes = 1280*2*36*28
        self.fc = nn.Linear(inplanes, 2)
    def forward(self, x):
        x = self.two_views_clf(x)
        if self.avgpool_flag:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# IMPLEMENTA FUNCAO {media, OU, etc} ENTRE AS DUAS PERNAS CC e MLO
class SideFunctionBreastModel(nn.Module):
    def __init__(self, device, model_file, network):
        super(SideFunctionBreastModel, self).__init__()
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network)
        # self.fc = nn.Linear(2048*2, 2)
    def forward(self, x):
        print(x.shape)
        x = self.two_views_clf(x)
        print(x.shape, x)  # torch.Size([1, 4]) tensor([[-1.0210,  1.0837, -0.4706,  0.4218]], device='cuda:0')
        print(torch.log_softmax(x, 1))
        x_cc = x[:, 1]      # take both malignant predictions
        x_mlo = x[:, 3]
        print(x_mlo.shape, x_mlo, x_cc)
        a = torch.tensor([[x_cc, x_mlo]]).cuda(0)
        x = torch.mean(a)
        x = torch.tensor([[0, x]]).cuda(0)
        print(x.shape, x)
        print('FIX ME - precisa ajustar os valores das predicao antes da mean....')
        sys.exit(0)
        return x

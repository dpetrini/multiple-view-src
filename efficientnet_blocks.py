
import collections
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet, MBConvBlock


class EFBlocks(nn.Module):
    def __init__(self, image_size=None, inplanes=1792,
                 outplanes=2048, n_blocks=1, strides=1, **kwargs):
        super(EFBlocks, self).__init__()

        temp_model = EfficientNet.from_name('efficientnet-b0', num_classes=5)

        global_params = temp_model._global_params

        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        self.drop_connect_rate = global_params.drop_connect_rate
        self.n_blocks = n_blocks

        # Parameters for an individual model block
        BlockArgs = collections.namedtuple('BlockArgs', [
            'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
            'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

        # Original consagrado de todas publicações. Porém tem tamanho de 16 Mi params, 
        #  grande pois ef-b0 tem 4 mi. (usado nos estudos doutorado)
        # block_args = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides],
        #                        expand_ratio=2, input_filters=inplanes,
        #                        output_filters=outplanes, se_ratio=0.25, id_skip=True)

        # Vamos tentar variações reduzindo tamanho e mantendo resultado hopefully. 03-2026
        block_args = BlockArgs(num_repeat=1, kernel_size=3, stride=[2],
                               expand_ratio=1, input_filters=inplanes,
                               output_filters=outplanes, se_ratio=0.25, id_skip=True)
        print("Patch Based: Using reduced top layer configuration.")

        self._blocks = nn.ModuleList([])
        for _ in range(0, n_blocks):
            self._blocks.append(MBConvBlock(block_args, global_params, image_size=image_size))

        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(outplanes, outplanes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=outplanes, momentum=bn_mom, eps=bn_eps)
        # self.swish = MemoryEfficientSwish()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(global_params.dropout_rate/2)
        self.fc = nn.Linear(outplanes, 2)

    def forward(self, x):
        # print('Input to block', x.shape)

        for i, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate)
            # print('Output to block', i+1, x.shape, drop_connect_rate)

        # x = self._conv_head(x)
        # x = self.bn1(x)
        # x = self.swish(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


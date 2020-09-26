import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot


# helping functions:

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups, \
                    dilation=dilation, padding=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convoltion with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    """ Creation of basic block of resnet that use bottleneck method of
    having 1x1, 3x3, & 1x1convs"""
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None):

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_planes * (base_width / 64.)) * groups
        # when stride != 1, both self.conv2 and self.downsample layers downsample the input

        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_planes*self.expansion)
        self.bn3 = norm_layer(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


    class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                replace_stride_with_dilation=None, norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation =1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride should be none")


        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, \
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, \
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, \
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, out_planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer
        downsample=None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride=1

        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsmaple = nn.Sequential(
                        conv1x1(self.in_planes, out_planes * block.expansion, stride),
                        norm_layer(out_planes * block.expansion),
            )


        layers= []
        layers.append(block(self.in_planes, out_planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.in_planes = out_planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes, groups=self.groups, \
                                base_width=self.base_width, dilation = self.dilation,
                                norm_layer = norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



## Feature Pyramid Network

def agg_node(in_planes, out_planes):
    """
    2 conv layers - first layer does 256, 256 and second does 256 -> 128

    """
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )


def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU(),
    )


class I2D(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=False):
        super(I2D, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)
        self.agg4 = agg_node(256, 128)

        # Upshuffle Layers
        self.up1 = upshuffle(128, 128, 8)
        self.up2 = upshuffle(128, 128, 4)
        self.up3 = upshuffle(128, 128, 2)

        # depth prediction
        self.predict1 = smooth(512, 128)
        self.predict2 = predict(128, 1)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size()

        # bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # top_down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        d5, d4, d3, d2 = self.up1(self.agg1(p5)), self.up2(self.agg2(p4)), self.up3(self.agg4(p3)), self.agg4(p2)
        _, _, H, W = d2.size()
        vol = torch.cat( [F.upsample(d, size=(H,W), mode='bilinear') for d in [d5, d4, d3, d2] ], dim=1)

        return self.predict2( self.predict1(vol) )

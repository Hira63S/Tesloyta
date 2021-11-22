import torch
import torch.nn as nn

__all__ = ['regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032',
           'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes,kernel_size=3,stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = planes * self.expansion
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width//min(width,group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
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
    
    
class RegNet(nn.Module):
    
    def __init__(self, block, layers, widths, num_classes=1000, zero_init_residual=True,
                group_width=1, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(RegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False, False]
            
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.group_width = group_width
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer3 = self._make_layer(block, widths[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)
        
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
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample=None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *=stride
            stride = 1
        if stride != 1 or self.planes != planes:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes, stride),
                norm_layer(planes),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.group_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, group_width=self.group_width,
                                dilation = self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self.forward_impl(x)

        
def regnetx_002(**kwargs):
    return RegNet(Bottleneck, [1,1,4,7], [24,56,152,368], group_width=8, **kwargs)

def regnetx_004(**kwargs):
    return RegNet(Bottleneck, [1,2,7,12], [32,64,160,384], group_width=16, **kwargs)

def regnetx_006(**kwargs):
    return RegNet(Bottleneck, [1,3,5,7], [48,96,240,528], group_width=24, **kwargs)

def regnetx_008(**kwargs):
    return RegNet(Bottleneck, [1,3,7,5], [64,128,288,672], group_width=16, **kwargs)



    
    
    
    
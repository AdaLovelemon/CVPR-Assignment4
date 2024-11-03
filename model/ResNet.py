import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.model_utils import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout_rate=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
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
        x = self.dropout(x)
        x = self.fc(x)
        return x

def resnet18(num_classes=10, dropout_rate=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout_rate)

def resnet34(num_classes=10, dropout_rate=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dropout_rate)

def resnet50(num_classes=10, dropout_rate=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout_rate)

def resnet101(num_classes=10, dropout_rate=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, dropout_rate)

def resnet152(num_classes=10, dropout_rate=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, dropout_rate)


def get_ResNet(config):
    dataset_name = config['Dataset']['dataset_name']
    ResNet_type = config['Model']['ResNet']['ResNet_type']
    num_classes = config['Dataset'][dataset_name]['num_classes']
    dropout_rate = config['Model']['ResNet']['dropout_rate']
    weight_init = config['Model']['ResNet']['weight_init']

    print(f'Current model: {ResNet_type}')
    if ResNet_type == 'ResNet18':
        model = resnet18(num_classes, dropout_rate)
    elif ResNet_type == 'ResNet34':
        model = resnet34(num_classes, dropout_rate)
    elif ResNet_type == 'ResNet50':
        model = resnet50(num_classes, dropout_rate)
    elif ResNet_type == 'ResNet101':
        model = resnet101(num_classes, dropout_rate)
    elif ResNet_type == 'ResNet152':
        model = resnet152(num_classes, dropout_rate)
    else:
        raise ValueError('Error ResNet Model')
    
    print(f'Current weight init: {weight_init}')
    if weight_init == 'kaiming':
        model.apply(kaiming_init)
    elif weight_init == 'xavier':
        model.apply(xavier_init)
    elif weight_init == 'zero':
        model.apply(zero_init)
    elif weight_init == 'uniform':
        model.apply(uniform_init)
    elif weight_init == 'normal':
        model.apply(normal_init)
    elif weight_init == 'constant':
        model.apply(constant_init)
    elif weight_init == 'orthogonal':
        model.apply(orthogonal_init)    
    else: 
        raise ValueError('Error Weight Init')
    
    return model
    



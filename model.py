"""
Model architecture optimized for CIFAKE (32x32 / 64x64) and general image detection.
Includes:
1. ResNet18 adapted for 32x32 CIFAR resolution (high accuracy ~94-96% on CIFAKE)
2. Fast Custom CNN for ultra-lightweight training
"""

import torch
import torch.nn as nn
from torchvision import models

def build_cifake_resnet18(num_classes=2):
    """
    Standard ResNet-18 modified for 32x32 image inputs:
    Replaces initial 7x7 conv and maxpool with a 3x3 conv so small 32x32 features
    are not lost in aggressive downsampling.
    """
    model = models.resnet18(weights=None)
    # CIFAR-adapted first layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove aggressive initial maxpool for 32x32
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class SimpleCIFAKENet(nn.Module):
    """
    A fast, lightweight 4-block ConvNet that trains quickly on both CPU and GPU.
    """
    def __init__(self, num_classes=2):
        super(SimpleCIFAKENet, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(model_type="resnet18", num_classes=2):
    if model_type.lower() == "simple":
        return SimpleCIFAKENet(num_classes=num_classes)
    return build_cifake_resnet18(num_classes=num_classes)

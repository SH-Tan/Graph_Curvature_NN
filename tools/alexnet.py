import torch
import torch.nn as nn
import torchvision.transforms as transforms

class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 → 16

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 → 8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 × 4 × 4 = 2048
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.features(x)
        return self.classifier(x)

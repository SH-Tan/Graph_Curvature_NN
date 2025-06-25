import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AlexNet_NoPool(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3, 32, 32)):
        super(AlexNet_NoPool, self).__init__()
        
        self.activation = nn.ReLU(inplace=True)
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                              std=(0.2023, 0.1994, 0.2010))
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            self.activation,
            nn.Conv2d(64, 192, kernel_size=5, padding=2, stride=2),
            self.activation,
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            self.activation,
        )

        # Automatically determine flattened feature size
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            flattened_dim = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 4096),
            self.activation,
            nn.Linear(4096, 4096),
            self.activation,
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



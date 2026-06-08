import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2

class Transfer_learning_Cifar(nn.Module):

    def __init__(self, num_classes=10, **kwargs, ):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.augmentation = nn.Sequential(
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomRotation(degrees=36),
                        v2.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )

        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg16_encoding = vgg16.features[:17]


        for index, prams in enumerate(self.vgg16_encoding.children()):
            if index <= 16:
                for pram in prams.parameters():
                    pram.requires_grad = False
        
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):

        if self.training:
            x = self.augmentation(x)

        x = self.vgg16_encoding(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x 
    
    


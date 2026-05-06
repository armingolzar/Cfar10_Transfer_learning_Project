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
        self.vgg16_encoding = vgg16.features

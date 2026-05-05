import torch 
from torchvision import dataset, transforms
from torch.utils.data import DataLoader

def preparing_dataset(target_size : int = 64, batch_size : int = 32, num_workers : int = 2):

    transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))])
    
    train_dataset = dataset.CIFAR10(root="..\\data\\", train=True, download=True, transform=transform)
    test_dataset = dataset.CIFAR10(root="..\\data\\", train=False, download=True, transform=transform)

    train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
    )

    test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
    )


    return train_loader, test_loader


import torch
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


def get_dataset_config(dataset_name):
    configs = {
        "mnist": {
            "color_channels": 1,
            "pooling_kernels": [2, 2],
            "encoder_output_size": 7,
            "decoder_input_size": 7,
            "lr_g": 2e-4,
            "lr_d": 2e-5,
            "batch_size": 1024,
            "latent_dim": 64,
            "img_size": 28
        },
        "fashion-mnist": {
            "color_channels": 1,
            "pooling_kernels": [2, 2],
            "encoder_output_size": 7,
            "decoder_input_size": 7,
            "lr_g": 2e-4,
            "lr_d": 1e-5,
            "batch_size": 512,
            "latent_dim": 64,
            "img_size": 28
        }
    }
    return configs.get(dataset_name)


def load_data(dataset_name, batch_size=512, num_workers=4):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if dataset_name == "mnist":
        train = MNIST(root="./data", train=True, transform=data_transform, download=True)
    elif dataset_name == "fashion-mnist":
        train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os
import pickle
import datetime
import argparse
from utils import create_visualization_grid


class EncoderModule(nn.Module):
    """ 编码器基础模块：卷积 -> 批归一化 -> ReLU，用于提取图像特征并逐步下采样 """
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """ 编码器：将输入的图像压缩成高维特征向量 """
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        super().__init__()
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)

        
class DecoderModule(nn.Module):
    """ 解码器基础模块：转置卷积 (Deconvolution) -> 批归一化 -> 激活函数，用于将特征向量还原回图像尺寸 """
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    """ 解码器：从隐变量 z 还原图像 """
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE(nn.Module):
    def __init__(self, color_channels=1, pooling_kernels=[2, 2], encoder_output_size=7, n_latent_features=64):
        super().__init__()
        
        self.n_latent_features = n_latent_features
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        self.encoder = Encoder(color_channels, pooling_kernels, n_neurons_middle_layer)
        
        # VAE 的核心：两个全连接层分别预测隐分布的均值(mu) 和方差的对数(logvar)
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)

        # 解码前的全连接层，将隐变量 z 映射回编码器输出的维度
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        self.decoder = Decoder(color_channels, pooling_kernels, encoder_output_size)

    def _reparameterize(self, mu, logvar):
        """ VAE 不能直接从 N(mu, sigma) 采样，因为采样操作不可导。解决方法：从 N(0, 1) 采样 eps，计算 z = mu + eps * sigma """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * eps
        return z
    
    def _bottleneck(self, h):
        """ 瓶颈层：生成分布参数并采样 """
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar
        
    def sample(self, num_samples=64, device='cpu'):
        """ 从标准正态分布采样 z，通过解码器生成新图像 """
        z = torch.randn(num_samples, self.n_latent_features, device=device)
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self._bottleneck(h)
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE 损失函数 = 重建损失 (BCE) + KL 散度 (KLD)
        BCE: 让生成的图更像原图
        KLD: 让隐空间分布接近标准正态分布 N(0, 1)
        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD, BCE + KLD


def get_dataset_config(dataset_name):
    configs = {
        "mnist": {
            "color_channels": 1,
            "pooling_kernels": [2, 2],
            "encoder_output_size": 7,
            "lr": 5e-4,
            "batch_size": 512,
            "latent_dim": 64
        },
        "fashion-mnist": {
            "color_channels": 1,
            "pooling_kernels": [2, 2],
            "encoder_output_size": 7,
            "lr": 5e-4,
            "batch_size": 512,
            "latent_dim": 64
        },
        "cifar": {
            "color_channels": 3,
            "pooling_kernels": [4, 2],
            "encoder_output_size": 4,
            "lr": 1e-4,
            "batch_size": 512,
            "latent_dim": 128
        },
        "stl": {
            "color_channels": 3,
            "pooling_kernels": [4, 4],
            "encoder_output_size": 6,
            "lr": 1e-3,
            "batch_size": 128,
            "latent_dim": 256
        }
    }
    return configs.get(dataset_name)


def load_data(dataset_name, batch_size=512, num_workers=4):
    data_transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        train = MNIST(root="./data", train=True, transform=data_transform, download=True)
        test = MNIST(root="./data", train=False, transform=data_transform, download=True)
    elif dataset_name == "fashion-mnist":
        train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
        test = FashionMNIST(root="./data", train=False, transform=data_transform, download=True)
    elif dataset_name == "cifar":
        train = CIFAR10(root="./data", train=True, transform=data_transform, download=True)
        test = CIFAR10(root="./data", train=False, transform=data_transform, download=True)
    elif dataset_name == "stl":
        train = STL10(root="./data", split="unlabeled", transform=data_transform, download=True)
        test = STL10(root="./data", split="test", transform=data_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, epoch, history):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    samples_cnt = 0
    
    print(f"\n{'='*60}")
    print(f"Epoch: {epoch+1} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    for batch_idx, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)

        bce, kld, loss = model.loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        samples_cnt += inputs.size(0)

        if batch_idx % 20 == 0:
            print(f"[Batch {batch_idx}/{len(train_loader)}] "
                  f"Loss: {train_loss/samples_cnt:.4f} | "
                  f"BCE: {train_bce/samples_cnt:.4f} | "
                  f"KLD: {train_kld/samples_cnt:.4f}")

    avg_loss = train_loss / samples_cnt
    avg_bce = train_bce / samples_cnt
    avg_kld = train_kld / samples_cnt
    
    history["train_loss"].append(avg_loss)
    history["train_bce"].append(avg_bce)
    history["train_kld"].append(avg_kld)
    
    print(f"\n[Train Summary] Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KLD: {avg_kld:.4f}")
    
    return avg_loss, avg_bce, avg_kld


def validate(model, test_loader, device, epoch, history):
    model.eval()
    val_loss = 0
    val_bce = 0
    val_kld = 0
    samples_cnt = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            recon_batch, mu, logvar = model(inputs)
            bce, kld, loss = model.loss_function(recon_batch, inputs, mu, logvar)
            
            val_loss += loss.item()
            val_bce += bce.item()
            val_kld += kld.item()
            samples_cnt += inputs.size(0)

    avg_loss = val_loss / samples_cnt
    avg_bce = val_bce / samples_cnt
    avg_kld = val_kld / samples_cnt
    
    history["val_loss"].append(avg_loss)
    history["val_bce"].append(avg_bce)
    history["val_kld"].append(avg_kld)
    
    print(f"[Val Summary]   Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KLD: {avg_kld:.4f}")
    
    return avg_loss, avg_bce, avg_kld


def save_checkpoint(model, optimizer, epoch, history, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch}")


def save_history(history, save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(history, fp)


def fit(model, train_loader, test_loader, optimizer, device, epochs, save_dir):
    history = {
        "train_loss": [], 
        "val_loss": [],
        "train_bce": [],
        "train_kld": [],
        "val_bce": [],
        "val_kld": []
    }
    
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device, epoch, history)
        validate(model, test_loader, device, epoch, history)
    
    print(f"\n>>> Saving final checkpoint and creating visualizations...")
    save_checkpoint(model, optimizer, epochs, history, f"{save_dir}/checkpoint_final.pth")
    save_history(history, f"{save_dir}/history.pkl")
    create_visualization_grid(model, test_loader, device, save_dir, "final")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Results saved in '{save_dir}/' directory")
    print(f"{'='*60}\n")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train VAE on various datasets')
    
    parser.add_argument('--dataset', type=str, default='cifar', 
                       choices=['mnist', 'fashion-mnist', 'cifar', 'stl'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-select by dataset if not specified)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (auto-select by dataset if not specified)')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Latent dimension (auto-select by dataset if not specified)')
    parser.add_argument('--num_workers', type=int, default=16,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: output/{dataset})')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = get_dataset_config(args.dataset)
    
    batch_size = args.batch_size if args.batch_size else config["batch_size"]
    lr = args.lr if args.lr else config["lr"]
    latent_dim = args.latent_dim if args.latent_dim else config["latent_dim"]
    save_dir = args.save_dir if args.save_dir else f"output/{args.dataset}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Initializing VAE model for {args.dataset} dataset...")
    
    model = VAE(
        color_channels=config["color_channels"],
        pooling_kernels=config["pooling_kernels"],
        encoder_output_size=config["encoder_output_size"],
        n_latent_features=latent_dim
    )
    model.to(device)
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    
    train_loader, test_loader = load_data(args.dataset, batch_size=batch_size, num_workers=args.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(test_loader.dataset)}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}")
    
    history = fit(model, train_loader, test_loader, optimizer, device, args.epochs, save_dir)
    
    print("\n" + "="*60)
    print("Training Summary:")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Train BCE: {history['train_bce'][-1]:.4f}")
    print(f"Final Train KLD: {history['train_kld'][-1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

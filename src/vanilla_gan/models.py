import torch
import torch.nn as nn


class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
    
    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, latent_dim, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 512 * decoder_input_size * decoder_input_size)
        self.bn0 = nn.BatchNorm2d(512)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.m1 = GeneratorBlock(512, 256, stride=1)
        self.m2 = GeneratorBlock(256, 128, stride=pooling_kernels[1])
        self.m3 = GeneratorBlock(128, 64, stride=pooling_kernels[0])
        self.bottle = GeneratorBlock(64, color_channels, stride=1, activation="tanh")
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 512, self.decoder_input_size, self.decoder_input_size)
        out = self.relu0(self.bn0(out))
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class Discriminator(nn.Module):
    def __init__(self, color_channels, pooling_kernels, encoder_output_size):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        
        self.bottle = DiscriminatorBlock(color_channels, 64, stride=1, kernel=1, pad=0)
        self.m1 = DiscriminatorBlock(64, 128, stride=1, kernel=3, pad=1)
        self.m2 = DiscriminatorBlock(128, 256, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = DiscriminatorBlock(256, 512, stride=pooling_kernels[1], kernel=3, pad=1)
        
        n_neurons = 512 * encoder_output_size * encoder_output_size
        self.fc = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        out = out.view(out.size(0), -1)
        return self.fc(out)

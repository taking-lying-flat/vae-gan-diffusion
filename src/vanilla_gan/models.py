import torch
import torch.nn as nn


# --- 基础模块：生成器卷积块 ---
class GeneratorBlock(nn.Module):
    """
    生成器的基本构建块：转置卷积 -> 批归一化 -> 激活函数
    用于将特征图的尺寸放大（上采样）
    """
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        # 使用转置卷积（Fractionally-strided Convolution）进行上采样
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
    
    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


# --- 基础模块：判别器卷积块 ---
class DiscriminatorBlock(nn.Module):
    """
    判别器的基本构建块：卷积 -> 批归一化 -> LeakyReLU
    用于从图像中提取特征并降低空间维度（下采样）
    """
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class Generator(nn.Module):
    """
    生成器：将随机噪声（隐向量）转换为伪造图像
    """
    def __init__(self, latent_dim, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.latent_dim = latent_dim

        # 第一层：全连接层，将隐向量扩展为可以 reshape 成特征图的大小
        self.fc = nn.Linear(latent_dim, 512 * decoder_input_size * decoder_input_size)
        self.bn0 = nn.BatchNorm2d(512)
        self.relu0 = nn.ReLU(inplace=True)

        # 逐层上采样，增加宽度和高度，减少通道数
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
    """
    判别器：判断输入的图像是“真”的还是“生成器伪造”的
    """
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

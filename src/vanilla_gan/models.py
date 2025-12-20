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
        
        # 根据参数选择激活函数
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "tanh":
            # Tanh 通常用于生成器的最后一层，将输出像素值映射到 [-1, 1]
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
        # 标准卷积层
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        # GAN 判别器通常使用 LeakyReLU 防止梯度消失
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


# --- 生成器网络 ---
class Generator(nn.Module):
    """
    生成器：将随机噪声（隐向量）转换为伪造图像
    """
    def __init__(self, latent_dim, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.latent_dim = latent_dim
        
        # 第一层：全连接层，将隐向量扩展为可以 reshape 成特征图的大小
        # 比如从 100 维变成 512 * 4 * 4
        self.fc = nn.Linear(latent_dim, 512 * decoder_input_size * decoder_input_size)
        self.bn0 = nn.BatchNorm2d(512)
        self.relu0 = nn.ReLU(inplace=True)
        
        # 逐层上采样，增加宽度和高度，减少通道数
        self.m1 = GeneratorBlock(512, 256, stride=1)
        self.m2 = GeneratorBlock(256, 128, stride=pooling_kernels[1])
        self.m3 = GeneratorBlock(128, 64, stride=pooling_kernels[0])
        
        # 输出层：使用 Tanh 激活函数，输出通道为图像颜色通道（如 RGB=3）
        self.bottle = GeneratorBlock(64, color_channels, stride=1, activation="tanh")
    
    def forward(self, z):
        # 1. 输入隐向量 z (batch_size, latent_dim)
        out = self.fc(z)
        # 2. Reshape 为特征图形状 (batch_size, 512, H, W)
        out = out.view(-1, 512, self.decoder_input_size, self.decoder_input_size)
        out = self.relu0(self.bn0(out))
        # 3. 通过一系列上采样块
        out = self.m3(self.m2(self.m1(out)))
        # 4. 输出最终图像
        return self.bottle(out)


# --- 判别器网络 ---
class Discriminator(nn.Module):
    """
    判别器：判断输入的图像是“真”的还是“生成器伪造”的
    """
    def __init__(self, color_channels, pooling_kernels, encoder_output_size):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        
        # 输入层：通常是 1x1 卷积改变通道数
        self.bottle = DiscriminatorBlock(color_channels, 64, stride=1, kernel=1, pad=0)
        
        # 提取特征层：逐层减少空间维度，增加通道数
        self.m1 = DiscriminatorBlock(64, 128, stride=1, kernel=3, pad=1)
        self.m2 = DiscriminatorBlock(128, 256, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = DiscriminatorBlock(256, 512, stride=pooling_kernels[1], kernel=3, pad=1)
        
        # 计算进入全连接层之前的总神经元数量
        n_neurons = 512 * encoder_output_size * encoder_output_size
        # 输出层：输出一个标量，代表真假的概率分数（未经过 Sigmoid，通常搭配含 Sigmoid 的损失函数）
        self.fc = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        # 1. 输入图像 (batch_size, color_channels, H, W)
        # 2. 通过卷积块提取特征
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        # 3. 展平 (Flatten)
        out = out.view(out.size(0), -1)
        # 4. 全连接层输出判定值
        return self.fc(out)

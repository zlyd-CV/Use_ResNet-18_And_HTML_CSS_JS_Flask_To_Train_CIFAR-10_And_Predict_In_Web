import torch.nn as nn
import torch


# 残差块：2层3×3卷积+BN+ReLU
class BasicBlock(nn.Module):
    expansion = 1  # 残差块输出通道数与中间通道数的倍数（BasicBlock为1）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一层卷积：3×3，步长由下采样需求决定
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channels)  # 原论文中BN在卷积后、ReLU前
        self.ReLU = nn.ReLU(inplace=True)

        # 第二层卷积：3×3，步长固定为1（仅第一层控制下采样）
        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 维度不匹配时的1×1卷积（shortcut路径）

    def forward(self, x):
        identity = x  # 恒等映射初始值（shortcut路径）
        # 主路径：conv1→bn1→relu→conv2→bn2
        out = self.Conv1(x)
        out = self.BN1(out)
        out = self.ReLU(out)

        out = self.Conv2(out)
        out = self.BN2(out)

        # 若维度不匹配（通道数/尺寸不同），用downsample调整shortcut路径
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：主路径 + shortcut路径，最后ReLU（原论文激活在残差相加后）
        out += identity
        out = self.ReLU(out)

        return out


class ResNet_18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_18, self).__init__()
        self.in_channels = 64  # 初始卷积输出通道数（原论文适配CIFAR-10调整）

        # 第一层：3×3卷积（原论文CIFAR-10设定，替代ImageNet的7×7卷积）
        self.Conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(self.in_channels)
        self.Relu = nn.ReLU(inplace=True)

        # 无池化层：原论文CIFAR-10用卷积步长2下采样，避免池化丢失信息
        self.MaxPool = nn.Identity()  # 恒等占位，保持与ImageNet ResNet结构兼容

        # 残差阶段：4个阶段，每个阶段2个BasicBlock（共8个残差块）
        # 阶段1：特征图32×32，通道64（无下采样，步长1）
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # 阶段2：特征图32×16，通道128（下采样，步长2）
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # 阶段3：特征图16×8，通道256（下采样，步长2）
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # 阶段4：特征图8×4，通道512（下采样，步长2）
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化：将4×4特征图压缩为1×1（原论文CIFAR-10收尾方式）
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        # 10路全连接：CIFAR-10共10个类别
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建残差阶段：blocks个残差块，处理下采样和维度匹配"""
        downsample = None
        # 需下采样（stride≠1）或输入输出通道不匹配时，用1×1卷积调整shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块：可能包含下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion  # 更新输入通道数
        # 后续残差块：无下采样（步长1）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播：初始卷积→BN→ReLU→残差阶段→全局池化→全连接
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.MaxPool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.AvgPool(x)
        x = torch.flatten(x, 1)  # 展平为向量（batch_size, 512）
        x = self.fc(x)

        return x

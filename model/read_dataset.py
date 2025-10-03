import random
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


# 划分出(训练集,验证集)和(测试集)--在本次任务中无用
def split_dataset_train_and_test(dataset, test_rate=None, seed=42):
    assert 0 < test_rate < 1
    total_size = len(dataset)
    test_size = int(test_rate * total_size)
    train_and_validation_size = total_size - test_size
    # 固定种子划分，保证每个epoch划分结果一致
    train_and_validation_dataset, test_dataset = random_split(
        dataset, [train_and_validation_size, test_size], generator=torch.Generator().manual_seed(seed))
    return train_and_validation_dataset, test_dataset


# 划分出(训练集)和(验证集)
def split_dataset_train_and_validation(dataset, validate_rate=None):
    assert 0 < validate_rate < 1
    seed = random.randint(0, 32767)
    total_size = len(dataset)
    validate_size = int(validate_rate * total_size)
    train_size = total_size - validate_size
    # 随机种子划分，确保每个epoch划分结果不一致
    train_dataset, validate_dataset = random_split(
        dataset, [train_size, validate_size], generator=torch.Generator().manual_seed(seed))
    return train_dataset, validate_dataset


def get_dataloader(batch_size,validate_rate=0.2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_and_validation_data = datasets.CIFAR10(root='../data', train=True, transform=transform_train,download=True)
    test_data = datasets.CIFAR10(root='../data', train=False, transform=transform_test,download=True)
    train_data, validate_data = split_dataset_train_and_validation(train_and_validation_data, validate_rate=validate_rate)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, validate_loader, test_loader

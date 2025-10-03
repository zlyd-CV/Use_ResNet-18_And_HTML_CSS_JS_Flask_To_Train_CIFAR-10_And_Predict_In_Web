import torch
from torch.optim.lr_scheduler import ExponentialLR
from networks import ResNet_18
from train_validate_test import train_validate_test_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
MODEL_PATH = './ResNet_18.pth'  # 模型保存路径
OPTIMIZER_PATH= './optimizer.pth'  # 优化器保存路径
SCHEDULER_PATH = "./scheduler.pth"  # 学习率衰减保存路径
if __name__ == '__main__':
    model = ResNet_18()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ExponentialLR(optimizer, gamma=0.9)  # gamma是衰减因子，范围(0,1)
    train_validate_test_model(model,criterion,optimizer,scheduler,EPOCHS,BATCH_SIZE,DEVICE,MODEL_PATH,OPTIMIZER_PATH,SCHEDULER_PATH)







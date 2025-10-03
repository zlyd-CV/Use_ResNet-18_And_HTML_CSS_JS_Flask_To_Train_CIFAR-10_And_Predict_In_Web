import os
import torch
from read_dataset import get_dataloader
from tqdm import tqdm


def train_validate_test_model(model, criterion, optimizer, scheduler, epochs, batch_size, device, model_path,
                              optimizer_path, scheduler_path):
    # 初始化变量用于保存最佳模型
    best_model_weights = None
    best_optimizer_state = None
    best_scheduler_state = None
    best_test_loss = float('inf')  # 仅用测试集损失判断（或改为验证集损失更合理）

    # 【关键修复1：只划分一次数据，避免每个epoch数据波动】
    train_loader, validate_loader, test_loader = get_dataloader(batch_size=batch_size)
    print(f"数据初始化完成：训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(validate_loader.dataset)},"
          f"测试集大小: {len(test_loader.dataset)}")

    # 加载历史状态
    if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
        try:
            best_model_weights = torch.load(model_path, map_location=device)
            model.load_state_dict(best_model_weights)
            best_optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(best_optimizer_state)
            best_scheduler_state = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(best_scheduler_state)
            # 加载历史最佳损失（可选：如果之前保存过，可从文件读取）
            print(f"加载历史模型成功：{model_path}, 优化器：{optimizer_path}, 调度器：{scheduler_path}")
        except Exception as e:
            print(f"加载历史状态失败：{e}，将从头训练")
            best_model_weights = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_scheduler_state = scheduler.state_dict()

    for epoch in range(epochs):
        print(f"\n第{epoch + 1}轮:")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_pbar = tqdm(train_loader, desc=f'训练进度：{epoch + 1}/{epochs}', unit="batch", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            train_accuracy += (predicted == labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_accuracy /= len(train_loader)
        print(f"[训练集] 准确率：{train_accuracy:.4f}, 平均损失：{train_loss:.4f}")

        # 验证阶段
        model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0
        with torch.no_grad():
            val_pbar = tqdm(validate_loader, desc=f'验证进度：{epoch + 1}/{epochs}', unit="batch", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * labels.size(0)
                predicted = torch.argmax(outputs, dim=1)
                validation_accuracy += (predicted == labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        validation_loss /= len(validate_loader.dataset)
        validation_accuracy /= len(validate_loader)
        print(f"[验证集] 准确率：{validation_accuracy:.4f}, 平均损失：{validation_loss:.4f}")

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'测试进度', unit="batch", leave=False)
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                predicted = torch.argmax(outputs, dim=1)
                test_accuracy += (predicted == labels).sum().item() / labels.size(0)
                test_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        test_loss /= len(test_loader.dataset)
        test_accuracy /= len(test_loader)
        print(f"[测试集] 准确率：{test_accuracy:.4f}, 平均损失：{test_loss:.4f}")

        # 【关键修复2：修正最佳模型判断与日志】
        if test_loss < best_test_loss:
            print(f"测试集损失下降（{best_test_loss:.4f} → {test_loss:.4f}），保存当前模型")
            best_test_loss = test_loss
            # 保存当前最佳状态
            best_model_weights = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_scheduler_state = scheduler.state_dict()
        else:
            # 【关键修复3：确保回退到最佳状态】
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
                optimizer.load_state_dict(best_optimizer_state)
                scheduler.load_state_dict(best_scheduler_state)
                print(f"测试集损失上升，回退到最佳模型（最佳损失：{best_test_loss:.4f}）")
            else:
                print("无历史最佳状态可回退")

        print('-' * 50)
        # 调度器更新（基于回退后的状态）
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"第{epoch + 1}轮结束，当前学习率: {current_lr:.6f}")

    # 保存最终最佳状态
    torch.save(best_model_weights, model_path)
    torch.save(best_optimizer_state, optimizer_path)
    torch.save(best_scheduler_state, scheduler_path)
    print(f"训练结束，最佳模型已保存至 {model_path}")
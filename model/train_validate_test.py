import os
import torch
from read_dataset import get_dataloader
from tqdm import tqdm

def train_validate_test_model(model, criterion, optimizer, scheduler, epochs, batch_size, device, model_path, optimizer_path, scheduler_path):
    # 初始化变量用于保存最佳模型
    best_model_weights = None
    best_optimizer_state = None
    best_scheduler_state = None
    # 若要继续上次训练，需要手动指定初始化值
    best_train_loss = float('inf')
    best_validation_loss = float('inf')
    best_test_loss = float('inf')  # 记得保存上一轮的最优测试集损失

    if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
        try:
            # 加载模型
            best_model_weights = torch.load(model_path, map_location=device)
            model.load_state_dict(best_model_weights)
            # 加载优化器
            best_optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(best_optimizer_state)
            # 加载调度器
            best_scheduler_state = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(best_scheduler_state)
            print(f"加载历史模型成功：{model_path},加载历史优化器成功：{optimizer_path},加载调度器状态成功：{scheduler_path}")
        except Exception as e:
            print(f"加载历史状态失败：{e}，将从头训练")
            # 加载失败后重置，避免后续None报错
            best_model_weights = None
            best_optimizer_state = None
            best_scheduler_state = None

    for epoch in range(epochs):
        # 在每个epoch开始时，重新划分训练集和验证集,测试集保持不变
        train_loader, validate_loader, test_loader = get_dataloader(batch_size=batch_size,validate_rate=0.2)
        print(f"\n第{epoch + 1}轮: 训练集大小: {len(train_loader)}, 验证集大小: {len(validate_loader)},"
              f"测试集大小: {len(test_loader)}")

        # 训练阶段
        model.train()  # 设置模型为训练模式
        train_loss = 0.0
        train_accuracy = 0.0
        train_pbar = tqdm(train_loader, desc=f'训练进度：{epoch + 1}/{epochs}', unit="batch",leave=False)  # 创建进度条对象
        for index, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算一个batch的平均损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            train_loss += loss.item()*images.size(0)
            predicted = torch.argmax(outputs.data, dim=1)  # 直接取第二个维度的最大值索引
            batch_correct = (predicted == labels).sum().item()  # 当前batch正确数
            batch_acc = batch_correct / labels.size(0)  # 当前batch正确率
            train_accuracy += batch_acc
            # 进度条显示当前batch的损失和正确率
            train_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_acc": f"{batch_acc:.4f}"
            })

        train_loss /= len(train_loader.dataset)  # 平均损失(累加损失除以样本总数)
        train_accuracy /= len(train_loader)
        print(f"\n[训练集：{epoch + 1}/{epochs}]——准确率：{train_accuracy:.4f};单个样本平均损失：{train_loss:.4f}")

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        validation_loss = 0.0
        validation_accuracy = 0.0
        with torch.no_grad():
            validation_pbar = tqdm(validate_loader, desc=f'验证进度：{epoch + 1}/{epochs}', unit="batch",leave=False)  # 创建进度条对象
            for index, (images, labels) in enumerate(validation_pbar):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                validation_loss += loss.item()*labels.size(0)
                predicted = torch.argmax(outputs.data, dim=1)  # 直接取第二个维度的最大值索引（即类别数）
                batch_correct = (predicted == labels).sum().item()  # 当前batch正确数
                batch_acc = batch_correct / labels.size(0)  # 当前batch正确率
                validation_accuracy += batch_acc
                # 进度条显示当前batch的损失和正确率
                validation_pbar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "batch_acc": f"{batch_acc:.4f}"
                })

        validation_loss /= len(validate_loader.dataset)  # 平均损失(累加损失除以样本总数)
        validation_accuracy /= len(validate_loader)
        print(f"\n[验证集：{epoch + 1}/{epochs}]——准确率：{validation_accuracy:.4f};单个样本平均损失：{validation_loss:.4f}")

        model.eval()  # 设置模型为评估模式(虽然上面已设置)
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'测试进度', unit="batch",leave=False)  # 创建进度条对象
            for index, (images, labels) in enumerate(test_pbar):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                test_loss += loss.item()*labels.size(0)
                predicted = torch.argmax(outputs.data, dim=1)  # 直接取第二个维度的最大值索引（即类别数）
                batch_correct = (predicted == labels).sum().item()  # 当前batch正确数
                batch_acc = batch_correct / labels.size(0)  # 当前batch正确率
                test_accuracy += batch_acc
                # 进度条显示当前batch的损失和正确率
                test_pbar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "batch_acc": f"{batch_acc:.4f}"
                })

            test_loss /= len(test_loader.dataset)  # 平均损失(累加损失除以样本总数)
            test_accuracy /= len(test_loader)
            print(f"[测试集：{epoch + 1}/{epochs}]——准确率：{test_accuracy:.4f};单个样本平均损失：{test_loss:.4f}")

        if test_loss < best_test_loss:
            print(f"测试集损失下降（{best_test_loss:.4f} → {test_loss:.4f}），保存当前模型")
            # 更新最佳损失
            best_train_loss = train_loss
            best_validation_loss = validation_loss
            best_test_loss = test_loss
            # 保存当前模型状态和优化器状态
            best_model_weights = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_scheduler_state = scheduler.state_dict()
            torch.save(best_model_weights, model_path)
            torch.save(best_optimizer_state, optimizer_path)
            torch.save(best_scheduler_state, scheduler_path)
        else:
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
                optimizer.load_state_dict(best_optimizer_state)
                scheduler.load_state_dict(best_scheduler_state)
                print(
                    f"损失上升，回退到上一次最佳模型：上一次训练损失: {best_train_loss:.4f}, "
                    f"验证损失: {best_validation_loss:.4f}, 测试损失：: {best_test_loss:.4f}, ")
            else:
                # 无最佳状态可回退（如首次训练）
                print(f"损失上升，但无历史最佳状态可回退（跳过回退）")
        # 调度器更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"第{epoch + 1}轮训练结束，当前学习率: {current_lr:.6f}")

        print('-' * 100)

        if best_model_weights is  None:
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scheduler.state_dict(), scheduler_path)  # 新增

import os
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from model.networks import ResNet_18

app = Flask(__name__)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
MODEL_PATH = './model/ResNet_18.pth'  # 确保此路径相对于 app.py 正确
NUM_CLASSES = 10  # CIFAR-10 有 10 个类别

# CIFAR-10 类别标签
CIFAR10_CLASSES = [
    '飞机', '汽车', '鸟', '猫猫', '鹿',
    '狗狗', '青蛙', '马', '船', '卡车'
]

# 用于推理的图像转换
transform_inference = transforms.Compose([
    transforms.Resize(32),  # 调整大小为 32x32，适用于 CIFAR-10
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = ResNet_18().to(device)  # 初始化模型结构
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
        model.eval()  # 设置模型为评估模式
        print(f"模型从 {MODEL_PATH} 加载成功")
    except Exception as e:
        print(f"从 {MODEL_PATH} 加载模型时出错: {e}")
        # 处理错误，可能会退出或在没有模型的情况下运行
else:
    print(f"未找到模型文件 {MODEL_PATH}。请确保模型已训练并保存。")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件部分'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    if file:
        try:
            image = Image.open(file.stream).convert('RGB')
            transformed_image = transform_inference(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(transformed_image)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                predicted_class_idx = predicted.item()
                predicted_class_name = CIFAR10_CLASSES[predicted_class_idx]

            return jsonify({'prediction': predicted_class_name})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__))) # 如果从不同目录运行，此行可能有用
    app.run(port=5001, debug=True)  # debug=True 允许在开发过程中自动重载和显示更好的错误信息，端口可自由定义

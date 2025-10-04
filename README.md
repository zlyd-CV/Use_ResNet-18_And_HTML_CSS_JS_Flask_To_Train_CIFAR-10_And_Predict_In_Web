# Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web

## 一、项目介绍
+ 本项目基于Pytorch构建了一个ResNet-18模型用于训练CIFAR数据集，考虑到数据集尺寸，用3×3卷积层替代了7×7卷积与池化层。
+ 本项目基于HTML+CSS+JavaScript前端和Flask后端实现简单的模型部署。
+ 论文图和模型示意图：
  
![模型示意图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/bf00a4872a1813874261b61b4d0f2af3c1ab7c72/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/Res-18%E7%BB%93%E6%9E%84%E5%9B%BE.png)

![原论文结构图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/bf00a4872a1813874261b61b4d0f2af3c1ab7c72/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/Res-18%E5%8E%9F%E8%AE%BA%E6%96%87%E7%BB%93%E6%9E%84%E5%9B%BE.png)
+ 项目流程图：
  
![项目流程图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/main/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/ResNet-50%E9%A2%84%E6%B5%8BCIFAR-10%E6%95%B0%E6%8D%AE%E9%9B%86.drawio.drawio.svg)

## 二、内容介绍
+ 目录结构
```python
├── .idea
├── app.py  # Flask框架生成本地Web端程序
├── data
│   ├── cifar-10-batches-py
│   ├── cifar-10-python.tar.gz
├── model
│   ├── main.py  # 训练模型程序
│   ├── networks.py
│   ├── optimizer.pth  # 优化器字典参数
│   ├── read_dataset.py
│   ├── ResNet_18.pth  # 模型字典参数参数
│   ├── scheduler.pth  # 学习率字典参数
│   ├── train_validate_test.py
├── photos
├── requirement.txt
├── static
│   ├── css
│   │   ├── style.css
│   ├── js
│   │   ├── script.js
├── templates
│   ├── index.html
├── test_version.py  # 检查包的版本
```
+ 本项目包含：
+ requirements.txt：包的版本，运行下面命令即可下载到虚拟环境中，pytorch请前往官网下载
 ```txt
pip install -r requirement.txt
```
+ model目录：模型训练，只需运行main.py即可训练
  该目录下的3个.pth文件对应以及训练好的模型、优化器、学习率控制器的字典参数，若你想从0开始训练请务必删掉它们
+ app.py：flask生成本地网页端主程序，若你已经有ResNet_18.pth，运行该文件即可生成本地网页端
+ data目录：数据下载地址，由于过大已被删除，运行model里的main.py会自动下载
+ static、templates目录：HTML+CSS+JS的存放目录
+ photos目录：一些论文图和可用来在浏览器里预测的图像

## 三、运行展示
![Web端效果展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/bf00a4872a1813874261b61b4d0f2af3c1ab7c72/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-10-03%20215947.png)
![Web端效果展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/bf00a4872a1813874261b61b4d0f2af3c1ab7c72/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-10-03%20220053.png)
![Web端效果展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/bf00a4872a1813874261b61b4d0f2af3c1ab7c72/Use_ResNet-18_And_HTML_CSS_JS_Flask_To_Train_CIFAR-10_And_Predict_In_Web/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-10-03%20220112.png)

## 四、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda
+ 原论文地址：[He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

## 五、未来考虑更新的方向
+ 增加更多经典卷积神经网络，例如LeNet、AlexNet、VGG、GoodLeNet等

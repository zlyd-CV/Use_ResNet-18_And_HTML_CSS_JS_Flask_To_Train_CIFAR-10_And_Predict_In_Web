document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadedImage = document.getElementById('uploadedImage');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');
    const loadingSpinner = document.getElementById('loadingSpinner');

    let selectedFile = null; // 用于存储文件对象

    // 处理图像文件选择
    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block';
                imagePlaceholder.style.display = 'none'; // 隐藏占位符
                predictButton.disabled = false; // 启用预测按钮
                predictionResult.textContent = '图像已准备好进行预测！';
                predictionResult.style.color = '#333'; // 重置颜色
            };
            reader.readAsDataURL(selectedFile);
        } else {
            uploadedImage.style.display = 'none';
            uploadedImage.src = '#';
            imagePlaceholder.style.display = 'block'; // 显示占位符
            predictButton.disabled = true; // 禁用预测按钮
            predictionResult.textContent = '上传图像并点击预测！';
            predictionResult.style.color = '#d9534f';
        }
    });

    // 处理预测按钮点击
    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('请先选择一张图片！');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        loadingSpinner.style.display = 'block'; // 显示加载旋转器
        predictButton.disabled = true; // 预测期间禁用按钮
        predictionResult.textContent = '正在预测...';
        predictionResult.style.color = '#3498db'; // 加载时显示蓝色

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP 错误！状态码: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                predictionResult.textContent = `错误: ${data.error}`;
                predictionResult.style.color = '#d9534f'; // 错误时显示红色
            } else {
                predictionResult.textContent = `预测类别: ${data.prediction.toUpperCase()}`;
                predictionResult.style.color = '#28a745'; // 成功时显示绿色
            }
        } catch (error) {
            console.error('预测失败:', error);
            predictionResult.textContent = `预测失败: ${error.message}`;
            predictionResult.style.color = '#d9534f'; // 错误时显示红色
        } finally {
            loadingSpinner.style.display = 'none'; // 隐藏加载旋转器
            predictButton.disabled = false; // 重新启用按钮
        }
    });
});
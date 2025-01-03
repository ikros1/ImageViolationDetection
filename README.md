# 图像分类与处理工具

## 介绍
该工具是一个基于PyTorch和Transformers库的图像分类与处理程序。它使用预训练的ViT模型来检测图像是否包含不适宜公开的NSFW（Not Safe For Work）内容，并将图像分类为“nsfw”或“normal”，然后将它们移动到相应的文件夹中。

## 功能
- **图像分类**：使用预训练的ViT模型检测图像是否包含NSFW内容。
- **图像转换**：将图像调整png格式，并重命名。
- **图像打包**：将图像按50个一组保存。

## 特性
- **多线程处理**：利用多线程提高处理速度。
- **日志记录**：记录处理过程中的信息和错误。
- **进度条**：显示处理进度。

## 安装依赖
在运行此项目之前，请确保您的系统上安装了以下依赖项：
此项目需要计算机配置ffmpeg
```bash
pip install -r requirements.txt
```
## 使用方法

### 1. 把待分类的图片放到img_path

### 2. 运行程序
将上述配置文件夹路径设置好后，直接运行脚本：
```bash
python start.py
```
### 3. 查看日志
日志文件会生成在当前目录下，文件名为`image_processing.log`。您可以查看该文件以获取处理过程中的详细信息。

### 4. 收取打包分类的成品
分类转换为png并重命名打包好的图片文件夹在target文件夹下，nsfw为工作不适宜图片，normal为正常图片

## 代码结构
- `run.py`：主脚本文件。
- `model/`：存放序列化的模型和处理器文件。
- `img_path/`：存放待处理的图像文件。
- `nsfw/`：存放检测为NSFW的图像文件。
- `normal/`：存放检测为正常的图像文件。

## 配置参数
- `MAX_WIDTH` 和 `MAX_HEIGHT`：设置图像的最大宽度和高度，默认为700像素。
- `num_workers`：设置使用的线程数，默认为系统CPU核心数。

## 示例
项目结构如下：
```
project/
├── script_name.py
├── model/
│   ├── model.pkl
│   └── processor.pkl
├── img_path/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── nsfw/
├── normal/
├── target
│   ├──normal
│   └──nsfw
└── image_processing.log
```

## 贡献
欢迎对本项目提出改进意见或修复bug。请提交pull request或创建issue。


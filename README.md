# 图像分类与处理工具

## 介绍
该工具是一个基于PyTorch和Transformers库的图像分类与处理程序。它使用预训练的ViT模型来检测图像是否包含不适宜公开的NSFW（Not Safe For Work）内容，并将图像分类为“nsfw”或“normal”，然后将它们移动到相应的文件夹中。

## 功能
- **图像分类**：使用预训练的ViT模型检测图像是否包含NSFW内容。
- **图像调整**：将图像调整为最大宽度和高度不超过700像素，同时保持长宽比例不变。
- **多线程处理**：利用多线程提高处理速度。
- **日志记录**：记录处理过程中的信息和错误。
- **进度条**：显示处理进度。

## 安装依赖
在运行此项目之前，请确保您的系统上安装了以下依赖项：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置文件夹路径
- `folder_path`：指定要处理的图像所在的文件夹路径。
- `target_folder1` 和 `target_folder2`：分别指定NSFW图像和正常图像的目标文件夹路径。

### 2. 运行程序
将上述配置文件夹路径设置好后，直接运行脚本：
```bash
python script_name.py
```

### 3. 查看日志
日志文件会生成在当前目录下，文件名为`image_processing.log`。您可以查看该文件以获取处理过程中的详细信息。

## 代码结构
- `script_name.py`：主脚本文件。
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
└── image_processing.log
```

运行脚本后，`img_path`文件夹中的图像将被分类并移动到`nsfw`或`normal`文件夹中。

## 贡献
欢迎对本项目提出改进意见或修复bug。请提交pull request或创建issue。

## 许可证
本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。
```

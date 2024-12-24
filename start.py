import shutil
import os
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from tqdm import tqdm
import logging
import time
import queue
import threading
import pickle

# 配置日志记录
log_file = 'image_processing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 指定目标文件夹路径
target_folder1 = 'nsfw'
target_folder2 = 'normal'

# 确保目标文件夹存在
os.makedirs(target_folder1, exist_ok=True)
os.makedirs(target_folder2, exist_ok=True)

# 指定文件夹路径
folder_path = 'img_path'
MAX_WIDTH = 700
MAX_HEIGHT = 700

# 列出文件夹下的所有文件和文件夹
files_and_folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     os.path.isfile(os.path.join(folder_path, f))]

# 序列化文件路径
model_file = 'model/model.pkl'
processor_file = 'model/processor.pkl'


# 加载模型和处理器
def load_model_processor():
    try:
        if os.path.exists(model_file) and os.path.exists(processor_file):
            with open(model_file, 'rb') as mf, open(processor_file, 'rb') as pf:
                model = pickle.load(mf)
                processor = pickle.load(pf)
            logging.info("模型和处理器从本地文件加载成功")
        else:
            model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
            processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
            with open(model_file, 'wb') as mf, open(processor_file, 'wb') as pf:
                pickle.dump(model, mf)
                pickle.dump(processor, pf)
            logging.info("模型和处理器加载成功并已序列化保存到本地")
    except Exception as e:
        logging.error(f"加载模型或处理器时出错: {e}")
        exit()
    return model, processor


model, processor = load_model_processor()


def detect(img):
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label.get(predicted_label, "未知标签")

    # 假设标签为'nsfw'表示不合规
    if predicted_class == 'nsfw':
        return True
    else:
        return False


# 创建任务队列
task_queues = []
num_workers = os.cpu_count()
for _ in range(num_workers):
    task_queues.append(queue.Queue())


# 文件分配线程
def file_distributor(files, task_queues):
    global all_files_processed
    all_files_processed = False
    while not all_files_processed:
        if not files:
            all_files_processed = True
            break
        file_name = files.pop(0)  # 从文件列表中弹出一个文件
        placed = False
        for q in task_queues:
            if q.empty():  # 找到第一个空闲的任务队列
                q.put(file_name)  # 将文件放入该队列
                placed = True
                progress_bar.update(1)  # 更新进度条
                break
        if not placed:
            files.append(file_name)  # 如果没有找到空闲队列，将文件重新放回列表末尾
        time.sleep(0.001)  # 防止CPU占用过高

# 处理文件的函数
def process_file(task_queue):
    while True:
        file_name = task_queue.get()
        if file_name is None:  # 结束信号
            break
        try:
            with open(file_name, 'rb') as file:
                img = Image.open(file).convert('RGB')  # 确保图像模式为RGB

                # 获取原始图像的宽度和高度
                width, height = img.size

                # 根据最大宽度和高度调整图像大小，保持长宽比例不变
                if width > MAX_WIDTH or height > MAX_HEIGHT:
                    ratio = min(MAX_WIDTH / width, MAX_HEIGHT / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                result = detect(img)
                file.close()  # 关闭文件句柄后再进行文件移动操作

                # 确定目标文件夹
                target_folder = target_folder1 if result else target_folder2

                # 构建目标文件路径
                base_name = os.path.basename(file_name)
                target_path = os.path.join(target_folder, base_name)

                # 检查目标文件夹是否存在
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder, exist_ok=True)
                    logging.warning(f"目标文件夹 {target_folder} 不存在，已创建")

                # 移动文件
                shutil.move(file_name, target_path)

                # logging.info(f"文件 {file_name} 被移动到 {target_path}")

        except FileNotFoundError:
            logging.error(f"文件 {file_name} 未找到")
        except PermissionError as e:
            logging.warning(f"处理文件 {file_name} 时出错: {e}。稍后重试...")
            task_queue.put(file_name)  # 将文件重新放回队列
        except Exception as e:
            logging.error(f"处理文件 {file_name} 时出错: {e}")

        task_queue.task_done()


if __name__ == '__main__':
    logging.info(f"使用 {num_workers} 个线程进行处理")

    # 初始化进度条
    progress_bar = tqdm(total=len(files_and_folders), desc="Processing images")

    # 创建文件分配线程
    distributor_thread = threading.Thread(target=file_distributor, args=(files_and_folders, task_queues))
    distributor_thread.start()

    # 创建处理线程
    threads = []
    for i in range(num_workers):
        thread = threading.Thread(target=process_file, args=(task_queues[i],))
        threads.append(thread)
        thread.start()

    # 等待所有任务完成
    for q in task_queues:
        q.join()

    # 发送结束信号给处理线程
    for _ in range(num_workers):
        for q in task_queues:
            q.put(None)

    # 等待所有处理线程完成
    for thread in threads:
        thread.join()

    distributor_thread.join()

    progress_bar.close()  # 关闭进度条
    logging.info("所有文件处理完毕")

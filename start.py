from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import queue
import threading
import pickle
import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import os
import shutil
import random
import datetime
from tqdm import tqdm

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


def delete_files_in_directory(directory):
    # 获取所有文件的列表
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_list.append(os.path.join(root, filename))

    if not file_list:
        print("指定目录下没有文件。")
        return

    # 提示用户确认删除操作
    confirmation = input(f"确定要删除 {len(file_list)} 个文件吗？(y/n): ")
    if confirmation.lower() != 'y':
        print("操作已取消。")
        return

    # 使用 tqdm 添加进度条
    with tqdm(total=len(file_list), desc="Deleting files", unit="file") as pbar:
        for file_path in file_list:
            try:
                os.remove(file_path)
                pbar.update(1)
            except Exception as e:
                print(f"无法删除文件 {file_path}: {e}")


def remove_spaces_in_filenames(directory):
    # 获取所有文件的列表
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if ' ' in filename:
                file_list.append((root, filename))

    # 使用 tqdm 添加进度条
    with tqdm(total=len(file_list), desc="Renaming files", unit="file") as pbar:
        for root, filename in file_list:
            old_file_path = os.path.join(root, filename)
            new_filename = filename.replace(' ', '_')  # 使用下划线代替空格
            new_file_path = os.path.join(root, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            pbar.update(1)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def convert_image(input_path, output_path):
    try:
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-pix_fmt', 'yuv420p', '-color_range', 'pc', output_path],
            check=True,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert {input_path} to {output_path}: {e.stderr.decode()}")
        return False
    return True


def process_images_in_thread(image_files, from_folder, target_folder, thread_id, progress_queue):
    for image_file in image_files:
        input_path = os.path.join(from_folder, image_file)
        timestamp = int(time.time())
        random_number = random.randint(1000, 9999)
        output_filename = f"{timestamp}_{random_number}.png"
        output_path = os.path.join(target_folder, output_filename)

        if not convert_image(input_path, output_path):
            print(f"Error processing {image_file}")
        progress_queue.put(1)


def process_images(from_folder, target_folder):
    create_folder_if_not_exists(target_folder)
    supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(from_folder) if f.lower().endswith(supported_formats)]

    num_cores = multiprocessing.cpu_count()
    chunk_size = len(image_files) // num_cores + (1 if len(image_files) % num_cores else 0)
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    progress_queue = multiprocessing.Queue()
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_images_in_thread, chunk, from_folder, target_folder, thread_id, progress_queue)
            for thread_id, chunk in enumerate(chunks, start=1)
        ]

        total_tasks = len(image_files)
        with tqdm(total=total_tasks, desc=f"Processing {from_folder}") as pbar:
            completed_tasks = 0
            while completed_tasks < total_tasks:
                if not progress_queue.empty():
                    completed_tasks += progress_queue.get()
                    pbar.update(1)


def batch_move_files(source_dir, target_dir, batch_size=50):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取源目录中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 计算总批次数
    total_batches = (len(files) + batch_size - 1) // batch_size

    # 使用 tqdm 显示进度条
    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]

            # 创建一个新的子目录，使用时间戳和随机数命名
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            random_number = random.randint(1000, 9999)
            batch_dir_name = f"batch_{timestamp}_{random_number}"
            batch_dir_path = os.path.join(target_dir, batch_dir_name)
            os.makedirs(batch_dir_path, exist_ok=True)

            # 将文件移动到新的子目录
            for file_name in batch_files:
                source_file_path = os.path.join(source_dir, file_name)
                target_file_path = os.path.join(batch_dir_path, file_name)
                shutil.move(source_file_path, target_file_path)

            # 更新进度条
            pbar.update(1)


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
    logging.info("所有文件分类完毕")

    from_folder_paths = ['nsfw', 'normal']  # 替换为你的文件夹路径
    target_folder_paths = ['from/nsfw_png', 'from/normal_png']

    for from_folder, target_folder in zip(from_folder_paths, target_folder_paths):
        remove_spaces_in_filenames(from_folder)
        process_images(from_folder, target_folder)
        delete_files_in_directory(from_folder)

    source_directory = 'from/normal_png'
    target_directory = 'target/normal'
    batch_move_files(source_directory, target_directory)
    source_directory = 'from/nsfw_png'
    target_directory = 'target/nsfw'
    batch_move_files(source_directory, target_directory)

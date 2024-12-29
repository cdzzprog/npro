import tkinter as tk
from tkinter import filedialog
import subprocess

# 创建主窗口
root = tk.Tk()
root.title("YOLOv8 训练界面")
root.geometry("400x300")

# 预设的默认参数
default_yaml_path = ""
default_model_path = ""  # 默认预训练模型路径
default_epochs = 50
default_imgsz = 640
default_batch = 4

# 创建全局变量
yaml_path = tk.StringVar(value=default_yaml_path)
model_path = tk.StringVar(value=default_model_path)

# 选择 YAML 配置文件
def select_yaml():
    yaml_file = filedialog.askopenfilename(title="选择数据集 YAML 文件", filetypes=[("YAML files", "*.yaml")])
    if yaml_file:
        yaml_path.set(yaml_file)

# 选择预训练模型
def select_model():
    model_file = filedialog.askopenfilename(title="选择预训练模型文件", filetypes=[("PyTorch Model files", "*.pt")])
    if model_file:
        model_path.set(model_file)

# 训练函数
def start_training():
    command = [
        "yolo", "train",
        f"data={yaml_path.get()}",
        f"model={model_path.get()}",
        f"epochs={default_epochs}",
        f"imgsz={default_imgsz}",
        f"batch={default_batch}",
    ]
    
    try:
        # 调用训练命令
        subprocess.run(command, check=True)
        print("训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")

# 创建选择 YAML 按钮
yaml_button = tk.Button(root, text="选择 YAML 文件", command=select_yaml)
yaml_button.pack(pady=10)

# 创建选择预训练模型按钮
model_button = tk.Button(root, text="选择预训练模型", command=select_model)
model_button.pack(pady=10)

# 显示选择的 YAML 文件路径
yaml_label = tk.Label(root, text="YAML 文件路径: " + yaml_path.get())
yaml_label.pack(pady=10)

# 显示选择的预训练模型路径
model_label = tk.Label(root, text="模型文件路径: " + model_path.get())
model_label.pack(pady=10)

# 创建开始训练按钮
train_button = tk.Button(root, text="开始训练", command=start_training)
train_button.pack(pady=20)

# 启动主循环
root.mainloop()

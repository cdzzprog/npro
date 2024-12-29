# import tkinter as tk
# from tkinter import filedialog, messagebox
# import subprocess
# import os

# # 默认类别名称
# classname = "landslide"

# # 创建主界面
# def create_ui():
#     # 创建窗口
#     window = tk.Tk()
#     window.title("YOLOv8 训练界面")
#     window.geometry("400x200")
    
#     # 标签
#     tk.Label(window, text="选择 YAML 文件").pack(pady=10)
#     yaml_path_entry = tk.Entry(window, width=40)
#     yaml_path_entry.pack(pady=5)
#     tk.Button(window, text="浏览", command=lambda: browse_yaml(yaml_path_entry)).pack(pady=5)
    
#     tk.Label(window, text="选择预训练模型").pack(pady=10)
#     model_path_entry = tk.Entry(window, width=40)
#     model_path_entry.pack(pady=5)
#     tk.Button(window, text="浏览", command=lambda: browse_model(model_path_entry)).pack(pady=5)
    
#     # 开始训练按钮
#     def start_training():
#         yaml_path = yaml_path_entry.get()
#         model_path = model_path_entry.get()
        
#         if not yaml_path or not model_path:
#             messagebox.showerror("错误", "请选择YAML文件和预训练模型")
#             return
        
#         # 执行训练脚本
#         try:
#             messagebox.showinfo("开始训练", "训练开始...")
#             train_model(yaml_path, model_path)
#         except Exception as e:
#             messagebox.showerror("错误", f"训练失败: {str(e)}")

#     tk.Button(window, text="开始训练", command=start_training).pack(pady=20)
    
#     # 进入事件循环
#     window.mainloop()

# # 浏览选择yaml文件
# def browse_yaml(entry):
#     filepath = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
#     if filepath:
#         entry.delete(0, tk.END)
#         entry.insert(0, filepath)

# # 浏览选择预训练模型
# def browse_model(entry):
#     filepath = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt")])
#     if filepath:
#         entry.delete(0, tk.END)
#         entry.insert(0, filepath)

# # 调用YOLOv8训练命令
# def train_model(yaml_path, model_path):
#     # 拼接命令
#     command = [
#         "yolo", "train", 
#         f"data={yaml_path}", 
#         f"model={model_path}",
#         f"epochs=50",  # 设置训练周期，可根据需求调整
#         f"imgsz=640",  # 设置图像大小，可根据需求调整
#         f"batch=16",  # 设置批次大小，可根据需求调整
#         f"name=landslide_train"  # 使用classname为默认类别名
#     ]
    
#     # 运行命令
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
#     if result.returncode == 0:
#         print("训练完成")
#         messagebox.showinfo("完成", "训练完成")
#     else:
#         print(f"训练失败: {result.stderr}")
#         messagebox.showerror("错误", f"训练失败: {result.stderr}")

# # 运行界面
# if __name__ == "__main__":
#     create_ui()
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import subprocess
# import os

# # 默认类别名称
# classname = "landslide"

# # 创建主界面
# def create_ui():
#     # 创建窗口
#     window = tk.Tk()
#     window.title("YOLOv8 训练界面")
#     window.geometry("400x200")
    
#     # 标签
#     tk.Label(window, text="选择 YAML 文件").pack(pady=10)
#     yaml_path_entry = tk.Entry(window, width=40)
#     yaml_path_entry.pack(pady=5)
#     tk.Button(window, text="浏览", command=lambda: browse_yaml(yaml_path_entry)).pack(pady=5)
    
#     tk.Label(window, text="选择预训练模型").pack(pady=10)
#     model_path_entry = tk.Entry(window, width=40)
#     model_path_entry.pack(pady=5)
#     tk.Button(window, text="浏览", command=lambda: browse_model(model_path_entry)).pack(pady=5)
    
#     # 开始训练按钮
#     def start_training():
#         yaml_path = yaml_path_entry.get()
#         model_path = model_path_entry.get()
        
#         if not yaml_path or not model_path:
#             messagebox.showerror("错误", "请选择YAML文件和预训练模型")
#             return
        
#         # 执行训练脚本
#         try:
#             messagebox.showinfo("开始训练", "训练开始...")
#             train_model(yaml_path, model_path)
#         except Exception as e:
#             messagebox.showerror("错误", f"训练失败: {str(e)}")

#     tk.Button(window, text="开始训练", command=start_training).pack(pady=20)
    
#     # 进入事件循环
#     window.mainloop()

# # 浏览选择yaml文件
# def browse_yaml(entry):
#     filepath = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
#     if filepath:
#         entry.delete(0, tk.END)
#         entry.insert(0, filepath)

# # 浏览选择预训练模型
# def browse_model(entry):
#     filepath = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt")])
#     if filepath:
#         entry.delete(0, tk.END)
#         entry.insert(0, filepath)

# # 调用YOLOv8训练命令
# def train_model(yaml_path, model_path):
#     # 拼接命令
#     command = [
#         "yolo", "train", 
#         f"data={yaml_path}", 
#         f"model={model_path}",
#         f"epochs=50",  # 设置训练周期，可根据需求调整
#         f"imgsz=640",  # 设置图像大小，可根据需求调整
#         f"batch=16",  # 设置批次大小，可根据需求调整
#         f"name=landslide_train"  # 使用classname为默认类别名
#     ]
    
#     # 运行命令并打印输出到控制台
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
#     # 打印标准输出和标准错误
#     print(result.stdout)  # 输出训练过程中的标准输出
#     print(result.stderr)  # 输出训练过程中的错误信息
    
#     if result.returncode == 0:
#         print("训练完成")
#         messagebox.showinfo("完成", "训练完成")
#     else:
#         print(f"训练失败: {result.stderr}")
#         messagebox.showerror("错误", f"训练失败: {result.stderr}")

# # 运行界面
# if __name__ == "__main__":
#     create_ui()
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

# 默认类别名称
classname = "landslide"

# 创建主界面
def create_ui():
    # 创建窗口
    window = tk.Tk()
    window.title("YOLOv8 训练界面")
    window.geometry("400x200")
    
    # 标签
    tk.Label(window, text="选择 YAML 文件").pack(pady=10)
    yaml_path_entry = tk.Entry(window, width=40)
    yaml_path_entry.pack(pady=5)
    tk.Button(window, text="浏览", command=lambda: browse_yaml(yaml_path_entry)).pack(pady=5)
    
    tk.Label(window, text="选择预训练模型").pack(pady=10)
    model_path_entry = tk.Entry(window, width=40)
    model_path_entry.pack(pady=5)
    tk.Button(window, text="浏览", command=lambda: browse_model(model_path_entry)).pack(pady=5)
    
    # 开始训练按钮
    def start_training():
        yaml_path = yaml_path_entry.get()
        model_path = model_path_entry.get()
        
        if not yaml_path or not model_path:
            messagebox.showerror("错误", "请选择YAML文件和预训练模型")
            return
        
        # 执行训练脚本
        try:
            messagebox.showinfo("开始训练", "训练开始...")
            train_model(yaml_path, model_path)
        except Exception as e:
            messagebox.showerror("错误", f"训练失败: {str(e)}")

    tk.Button(window, text="开始训练", command=start_training).pack(pady=20)
    
    # 进入事件循环
    window.mainloop()

# 浏览选择yaml文件
def browse_yaml(entry):
    filepath = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
    if filepath:
        entry.delete(0, tk.END)
        entry.insert(0, filepath)

# 浏览选择预训练模型
def browse_model(entry):
    filepath = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pt")])
    if filepath:
        entry.delete(0, tk.END)
        entry.insert(0, filepath)

# 调用YOLOv8训练命令
def train_model(yaml_path, model_path):
    # 拼接命令
    command = [
        "yolo", "train", 
        f"data={yaml_path}", 
        f"model={model_path}",
        f"epochs=50",  # 设置训练周期，可根据需求调整
        f"imgsz=640",  # 设置图像大小，可根据需求调整
        f"batch=16",  # 设置批次大小，可根据需求调整
     # 使用classname为默认类别名
    ]
    
    # 使用Popen启动进程以便实时获取输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 实时读取并打印标准输出和标准错误
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line, end="")  # 输出训练过程中的标准输出
    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line, end="")  # 输出训练过程中的标准错误

    # 等待命令执行完成
    process.stdout.close()
    process.stderr.close()
    process.wait()

    if process.returncode == 0:
        print("训练完成")
        messagebox.showinfo("完成", "训练完成")
    else:
        print(f"训练失败: {stderr_line}")
        messagebox.showerror("错误", f"训练失败: {stderr_line}")

# 运行界面
if __name__ == "__main__":
    create_ui()

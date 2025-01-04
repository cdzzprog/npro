import numpy as np
import os
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, IntVar, Label, messagebox
from PIL import Image, ImageTk,ImageEnhance, ImageDraw, ImageFont
import ctypes
import threading
import subprocess
import datetime
import time
import logging
from queue import Queue, Empty
import queue 
import random
from PIL import Image
from ultralytics import YOLO
class App(ctk.CTk):
 # 用于跟踪登录状态
    def sidebar_button_event(self):
        if not self.logged_in:
            messagebox.showerror("Error", "You must be logged in to use this feature.")
        #     return
        # messagebox.showinfo("Function 1", "Function 1 executed successfully.")
    def login(self, event):
        registered_users = {}
        # Registration function
        def register():
            username = entry_register_username.get()
            password = entry_register_password.get()
            if username in registered_users:
                messagebox.showerror("Error", "用户名已注册")
            else:
                registered_users[username] = password
                messagebox.showinfo("Success", "用户注册成功")
                root.deiconify() 
                register_frame.pack_forget()
                show_login()
        # Login function
        def login():
            username = entry_username.get()
            password = entry_password.get()
            if username in registered_users and registered_users[username] == password:
                root.deiconify() 
                messagebox.showinfo("Success", "登录成功")
                self.logged_in=True
                root.destroy()
            else:
                messagebox.showerror("Error", "用户名或密码错误")
        # Function to show the registration screen
        def show_register():
            login_frame.pack_forget()
            register_frame.pack(pady=20, padx=60, fill="both", expand=True)
        # Function to show the login screen
        def show_login():
            register_frame.pack_forget()
            login_frame.pack(pady=20, padx=60, fill="both", expand=True)
        root = ctk.CTk()
        root.geometry("500x500")
        root.title("高原地区山体滑坡遥感智能识别系统")
        # root.configure(bg="#F0F2F5")  # 设置背景颜色为浅灰色
        root.configure(bg="#2F3136")  # 设置窗口背景颜色为与框架一致的深色
        # 设置CTk的主题和样式
        ctk.set_appearance_mode("System")  # 设置主题为系统默认（可以自动切换暗/亮模式）
        ctk.set_default_color_theme("blue") 
        # 设置CTk的主题和样式
        ctk.set_appearance_mode("System")  # 设置主题为系统默认（可以自动切换暗/亮模式）
        ctk.set_default_color_theme("blue")  # 设置蓝色为默认主题颜色
        # 获取屏幕的宽度和高度
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # 计算窗口的左上角坐标，使其居中
        window_width = 500
        window_height = 600
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        # 设置窗口的起始位置
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        # 创建登录框架
        login_frame = ctk.CTkFrame(master=root, width=400, height=300, bg_color="#2F3136",border_width=0,corner_radius=0)
        login_frame.place(relx=0.5, rely=0.5, anchor="center")  # 使用place将框架放在窗口中心
        # 登录标题
        label_login = ctk.CTkLabel(master=login_frame, text="系统登录", font=("微软雅黑", 24, "bold"), text_color="white")
        label_login.pack(pady=20)
        # 用户头像
        label_avatar = ctk.CTkLabel(master=login_frame, text="👤", font=("微软雅黑", 40), text_color="white")
        label_avatar.pack(pady=10)
        # 用户名输入框
        entry_username = ctk.CTkEntry(master=login_frame, placeholder_text="请输入用户名", width=280, height=40, font=("微软雅黑", 14))
        entry_username.pack(pady=10)
        # 密码输入框
        entry_password = ctk.CTkEntry(master=login_frame, placeholder_text="请输入密码", show="*", width=280, height=40, font=("微软雅黑", 14))
        entry_password.pack(pady=10)
        # 登录按钮
        button_login = ctk.CTkButton(master=login_frame, text="登录", width=280, height=40, font=("微软雅黑", 14), command=login, corner_radius=8, fg_color="#4CAF50", hover_color="#81C784")
        button_login.pack(pady=10)
        # 注册按钮
        button_show_register = ctk.CTkButton(master=login_frame, text="注册", width=280, height=40, font=("微软雅黑", 14), command=show_register, corner_radius=8, fg_color="#2196F3", hover_color="#64B5F6")
        button_show_register.pack(pady=10)
        # "记住我"复选框
        checkbox_login = ctk.CTkCheckBox(master=login_frame, text="记住我", font=("微软雅黑", 12), text_color="white")
        checkbox_login.pack(pady=10)
        # 创建注册框架
        register_frame = ctk.CTkFrame(master=root, width=400, height=300, bg_color="#2F3136",corner_radius=0)
        label_register = ctk.CTkLabel(master=register_frame, text="系统注册", font=("微软雅黑", 24, "bold"), text_color="white")
        label_register.pack(pady=20)
        entry_register_username = ctk.CTkEntry(master=register_frame, placeholder_text="请输入用户名", width=280, height=40, font=("微软雅黑", 14))
        entry_register_username.pack(pady=10)
        entry_register_password = ctk.CTkEntry(master=register_frame, placeholder_text="请输入密码", show="*", width=280, height=40, font=("微软雅黑", 14))
        entry_register_password.pack(pady=10)
        # 注册按钮
        button_register = ctk.CTkButton(master=register_frame, text="注册", width=280, height=40, font=("微软雅黑", 14), command=register, corner_radius=8, fg_color="#2196F3", hover_color="#64B5F6")
        button_register.pack(pady=10)
        # 返回登录按钮
        button_show_login = ctk.CTkButton(master=register_frame, text="返回登录", width=280, height=40, font=("微软雅黑", 14), command=show_login, corner_radius=8, fg_color="#4CAF50", hover_color="#81C784")
        button_show_login.pack(pady=10)
        label_copyright = ctk.CTkLabel(master=root, text="(The Remote Sensing Intelligent Recognition System for Landslides in Plateau Regions)", font=("微软雅黑", 10), text_color="black")
        label_copyright.place(relx=0.5, rely=0.95, anchor="center")
        # 初始化时显示登录界面
        login_frame.place(relx=0.5, rely=0.5, anchor="center")
        root.mainloop()
    def get_screen_size(self):
        return self.winfo_screenwidth(), self.winfo_screenheight()
      # 判断用户名是否只包含数字
    def is_valid_username(username):
        return username.isdigit()
    # 判断密码是否合法（这里只允许字母和数字，不允许特殊字符）
    def is_valid_password(password):
        # 只允许字母和数字，其他特殊字符不允许
        return bool(re.match("^[A-Za-z0-9]*$", password))
    # 注册函数
    def registernew(username, password):
        # 判断用户名是否为空
        if not username.strip():
            return "用户名不能为空！"
        # 判断用户名是否符合要求（只能是数字）
        if not is_valid_username(username):
            return "用户名只能包含数字！"
        # 判断密码是否为空
        if not password.strip():
            return "密码不能为空！"
        # 判断密码是否合法（不能有非法字符）
        if not is_valid_password(password):
            return "密码包含非法字符，只能包含字母和数字！"
        # 判断用户名是否已被注册
        if username in user_data:
            return "该用户名已被注册！"
        # 注册成功
        user_data[username] = password
        return "注册成功！"
    # 登录函数
    def loginnew(username, password):
        if username not in user_data:
            return "用户名不存在！"
        elif user_data[username] != password:
            return "密码错误！"
        else:
            return "登录成功！"
    def get_yaml_path():
    # 获取当前执行的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, 'data.yaml')
        return yaml_path
    def get_model_path():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pt')
        return model_path
    def load_data(image_dir, label_file):
        df = pd.read_csv(label_file)
        image_paths = [os.path.join(image_dir, img_name) for img_name in df['image_name']]
        labels = df['label'].tolist()
        return image_paths, labels
    def split_data(image_paths, labels, test_size=0.2, val_size=0.1):
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=test_size + val_size, random_state=42
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42
        )
        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    def index_event(self):
        """
        跳转到主页事件处理函数
        """
        if hasattr(self, 'main1') and hasattr(self, 'main2'):
                self.main1.grid_forget()
                for widget in self.main1.winfo_children():
                    widget.destroy()
                self.main2.grid_forget()
                for widget in self.main1.winfo_children():
                    widget.destroy()
        else: 
        # 如果 main1 和 main2 没有创建，则清空 main
            if hasattr(self, 'main'):
                # self.main.grid_forget()
                for widget in self.main.winfo_children():
                    widget.destroy()
    def train_event(self):
        """
        跳转到训练事件处理函数，并刷新内容
        """
        # 清空当前的内容区域
        if not self.logged_in:
            messagebox.showerror("Error", "请登录后使用功能.")
        else:    
            for widget in self.main.winfo_children():
                widget.destroy()
            self.after(500,self.train_page())
    def select_yaml(self):
        yaml_file = filedialog.askopenfilename(title="选择数据集 YAML 文件", filetypes=[("YAML files", "*.yaml")])
        if yaml_file:
            self.yaml_path.set(yaml_file)
    # 选择预训练模型
    def select_model(self):
        model_file = filedialog.askopenfilename(title="选择预训练模型文件", filetypes=[("PyTorch Model files", "*.pt")])
        if model_file:
            self.model_path.set(model_file)
    def update_logs(self, message):
        """更新日志框中的文本"""
        self.logs.insert("end", message + "\n")
        self.logs.yview("end")  #
    def start_traininglast(self):
        # 启动一个线程来执行训练任务
        threading.Thread(target=self.start_trainingt, daemon=True).start()
    def start_trainingt(self):
        self.update_logs(f"训练开始，请等待...\n使用的yaml路径: {self.yaml_path.get()}\n使用的模型路径: {self.model_path.get()}")
        command = [
            "yolo", "train",
            f"data={self.yaml_path.get()}",
            f"model={self.model_path.get()}",
            f"epochs={self.default_epochs}",
            f"imgsz={self.default_imgsz}",
            f"batch={self.default_batch}",
        ]
        try:
            # 调用训练命令
            subprocess.run(command, check=True)
            self.update_logs("训练完成！")
            print("训练完成！")
        except subprocess.CalledProcessError as e:
            print(f"训练失败: {e}")
            self.update_logs(f"训练失败: {e}")
    def start_trainingos(self):
        command = f"yolo train data={self.yaml_path.get()} model={self.model_path.get()} epochs={self.default_epochs} imgsz={self.default_imgsz} batch={self.default_batch}"
        try:
            # 使用 os.system 执行命令，输出直接显示在终端
            return_code = os.system(command)
            if return_code == 0:
                self.after(500,self.logs.insert(tk.END, "\n训练完成！\n"))
            else:
                self.logs.insert(tk.END, "\n训练过程中发生错误！\n")
            self.logs.yview(tk.END)
        except Exception as e:
            self.logs.insert(tk.END, f"\n训练失败: {e}\n")
            self.logs.yview(tk.END)
    def start_training11(self):
        command = f"yolo train data={self.yaml_path.get()} model={self.model_path.get()} epochs={self.default_epochs} imgsz={self.default_imgsz} batch={self.default_batch}"
        try:
            # 使用 subprocess.Popen 执行命令并实时读取输出
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,encoding='utf-8')
            # 实时读取标准输出并更新到 logs
            for line in process.stdout:
                self.logs.insert(tk.END, line)
                self.logs.yview(tk.END)
            # 读取标准错误并更新到 logs
            for line in process.stderr:
                self.logs.insert(tk.END, f"错误: {line}")
                self.logs.yview(tk.END)
            # 等待命令执行完成并获取返回码
            process.wait()
            # 检查命令执行的状态
            if process.returncode == 0:
                self.logs.insert(tk.END, "\n训练完成！\n")
            else:
                self.logs.insert(tk.END, "\n训练过程中发生错误！\n")
            self.logs.yview(tk.END)
        except Exception as e:
            self.logs.insert(tk.END, f"\n训练失败: {e}\n")
            self.logs.yview(tk.END)
    def start_training(self):
        # 启动一个线程来执行训练任务
        threading.Thread(target=self.start_training_task, daemon=True).start()
        # 启动一个定时器以定期检查队列中的消息并更新UI
        self.main.after(100, self.process_queue)
# 训练任务执行的实际功能
    def start_training_task(self):
        # command = f"yolo train data={self.yaml_path.get()} model={self.model_path.get()} epochs={self.default_epochs} imgsz={self.default_imgsz} batch={self.default_batch}"
        command = [
            "yolo", "train",
            f"data={self.yaml_path.get()}",
            f"model={self.model_path.get()}",
            f"epochs={self.default_epochs}",
            f"imgsz={self.default_imgsz}",
            f"batch={self.default_batch}",
        ]
        print(command)
        print(self.yaml_path.get())
        print(self.model_path.get())
        try:
            # 使用 subprocess.Popen 执行命令并实时读取输出
            # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,, universal_newlines=True, encoding='utf-8',errors='replace')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8',errors='replace')
            # 实时读取标准输出并将其放入队列
            for line in process.stdout:
                self.queue.put(line)
            # 读取标准错误并将其放入队列
            for line in process.stderr:
                self.queue.put(f"错误: {line}")
            # 等待命令执行完成并获取返回码
            process.wait()
            # 检查命令执行的状态
            if process.returncode == 0:
                self.queue.put("\n训练完成！\n")
            else:
                self.queue.put("\n训练过程中发生错误！\n")
        except Exception as e:
            # 捕获任何异常并将错误信息放入队列
            self.queue.put(f"\n训练失败: {e}\n")
    def process_queue(self):
        try:
            # 从队列中获取并更新UI
            while True:
                message = self.queue.get_nowait()
                self.logs.insert(tk.END, message)
                self.logs.yview(tk.END)
        except queue.Empty:
            pass
        # 定期检查队列中的消息
        self.main.after(100, self.process_queue)
    def train_page(self):
        """
        加载数据处理内容的具体函数
        """
        # 数据预处理页面
        # self.default_yaml_path = r"路径"
        # self.default_model_path = r"路径"
        self.default_yaml_path = ""
        self.default_model_path = ""
        self.default_epochs = 10
        self.default_imgsz = 640
        self.default_batch = 1
        self.queue = queue.Queue()
        # 绑定变量
        self.yaml_path = tk.StringVar(value=self.default_yaml_path)
        self.model_path = tk.StringVar(value=self.default_model_path)
        self.button_frame = ctk.CTkFrame(self.main)
        self.button_frame.grid(row=0, column=0, padx=20, pady=10, sticky='nsew')
        # self.main.grid_columnconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        # self.main.grid_rowconfigure(0, weight=1)
        # self.main.grid_rowconfigure(1, weight=1)
        # self.main.grid_rowconfigure(2, weight=0)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        button_params = {
            "width": 120,  
            "height": 40,  
            "corner_radius": 10,  
            "font": ctk.CTkFont(size=14, weight="bold"),  
        }
        # 选择文件夹按钮 (绿色)
        self.datapre_button_1 = ctk.CTkButton(
            self.button_frame, 
            text="选择yaml文件", 
            command=self.select_yaml, 
            fg_color="#4CAF50",  
            hover_color="#45a049",  
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.datapre_button_1.grid(row=0, column=0, padx=20, pady=10)
        self.datapre_button_3 = ctk.CTkButton(
            self.button_frame, 
            text="开始训练", 
            command=self.start_training, 
            fg_color="#FF9800",  # 按钮背景颜色
            hover_color="#F57C00",  # 悬停时按钮的背景颜色
            border_color="#E65100",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_3.grid(row=0, column=1, padx=20, pady=10)
        # 上一个 (红色)
        self.datapre_button_2 = ctk.CTkButton(
            self.button_frame, 
            text="选择预训练模型", 
            command=self.select_model, 
            fg_color="#F44336",  # 按钮背景颜色
            hover_color="#D32F2F",  # 悬停时按钮的背景颜色
            border_color="#C62828",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_2.grid(row=0, column=2, padx=20, pady=10)
        self.logs = ctk.CTkTextbox(self.main, width=1000, height=900, fg_color="#E1E1E1")
        self.logs.grid(row=2, column=0, padx=20, pady=20)
    def load_model(self, model_path):
        """
        加载YOLOv8模型
        """
        try:
            # 加载YOLOv8模型
            self.model = YOLO(model_path)
            print(f"模型 {model_path} 加载成功！")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    def selectpremodel_folder(self):
        self.selected_model_var=tk.StringVar()
        file_path = filedialog.askopenfilename(
            title="选择模型",
            filetypes=[("PyTorch模型文件", "*.pt")]
        )
        if file_path:
            self.model_label.configure(text=f"已选择模型: {file_path}")
            self.selected_model_var.set(f"已选择模型")
            self.load_model(file_path)
            self.model_path = file_path
    def predict_with_yolopil(self, image_path):
        """
        使用YOLOv8模型进行图像预测，并返回带有检测框的图像。
        参数:
        - image_path (str): 图像文件的路径
        - model_path (str): YOLOv8模型的路径，默认为 "yolov8n.pt"
        返回:
        - PIL.Image.Image: 带有检测框的图像
        """
        # 加载YOLOv8模型
        model = YOLO(self.model_path)  # 你可以选择不同的模型版本，例如 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
        # 使用PIL打开图像并确保它是RGB模式
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        # 使用YOLOv8进行预测
        results = model(img_np)
        # 获取预测框信息
        predictions = results[0].boxes
        try:
            font = ImageFont.truetype("arial.ttf", size=30)  # 这里可以指定字体和大小
        except IOError:
            font = ImageFont.load_default() 
        # 创建一个ImageDraw对象，用于在图像上绘制框
        draw = ImageDraw.Draw(img)
        # 遍历所有预测框并在图像上绘制
        for pred in predictions:
            # 获取xywh坐标
            xywh = pred.xywh[0]  # 假设pred.xywh是一个包含4个元素的列表或张量
            # 安全地访问x, y, w, h
            if len(xywh) < 4:
                print("Invalid prediction, skipping.")
                continue  # 跳过不完整的预测框
            x, y, w, h = xywh[0].item(), xywh[1].item(), xywh[2].item(), xywh[3].item()
            conf, cls = pred.conf.item(), pred.cls.item()
            # 转换为图像坐标
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)
            # 绘制边界框
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            # 确保results是一个模型返回的对象，获取类别名称
            # print(f"Results type: {type(results)}")  # 查看返回的results的类型
            # print(f"Results content: {results}") 
            class_name = "landslide"  # 获取类别名称
            label = f"{class_name} {conf:.2f}"
            text_width, text_height = draw.textsize(label, font=font)
            text_x = left + (right - left) / 2 - text_width / 2
            text_y = top + (bottom - top) / 2 - text_height / 2
            # 在边界框旁边添加类别名和置信度
            draw.text((text_x, text_y), label, fill="red", font=font)
        # 返回修改后的图像（带检测框）
        return img        
    def predict_with_yolo(self, image_path):
        """
        使用YOLOv8模型进行图像预测，并返回带有检测框的图像。
        参数:
        - image_path (str): 图像文件的路径
        返回:
        - PIL.Image.Image: 带有检测框的图像
        """
        # 加载YOLOv8模型
        model = YOLO(self.model_path)  # 你可以选择不同的模型版本，例如 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
        # 使用PIL打开图像并确保它是RGB模式
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        # 使用YOLOv8进行预测
        results = model(img_np)
        # 获取预测结果（results[0] 是包含所有预测信息的 Results 对象）
        results = results[0]  # 获取第一个预测结果对象
        # 获取boxes（每个框的位置，xyxy格式）
        boxes = results.boxes.xyxy  # 返回框的四个坐标[xmin, ymin, xmax, ymax]
        # 创建一个ImageDraw对象，用于在图像上绘制框
        draw = ImageDraw.Draw(img)
        # 使用一个大的字体
        try:
            font = ImageFont.truetype("arial.ttf", size=30)  # 这里可以指定字体和大小
        except IOError:
            font = ImageFont.load_default()  # 如果无法加载指定字体，就使用默认字体
        # 遍历所有预测框并在图像上绘制
        for i, box in enumerate(boxes):
            # 获取当前框的坐标
            left, top, right, bottom = map(int, box)  # 获取左上角和右下角的坐标
            # 获取类别索引，并映射到类别名称
            # class_idx = int(results.names[i])  # 获取类别索引
            # class_name = results.names[class_idx]  # 通过索引获取类别名称
            class_name = results.names[int(results.boxes.cls[i])]
            conf = results.boxes.conf[i]  # 获取框的置信度
            label = f"{class_name} {conf:.2f}"  # 构建标签文本，包含类别名称和置信度
            # 计算标签文本的宽度和高度，以便将它居中显示
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
            # 计算文本位置（使其居中）
            # text_x = left + (right - left) / 2 - text_width / 2
            # text_y = top + (bottom - top) / 2 - text_height / 2
            text_x = left + (right - left) / 2 - text_width / 2
            text_y = top - text_height - 5
            # 绘制边界框
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            # 绘制文本（标签）
            draw.text((text_x, text_y), label, fill="red", font=font)
        # 返回修改后的图像（带检测框）
        return img
    def predict_with_yolocv2(self, image_path):
        try:
            # 加载YOLO模型（可以选择指定权重文件路径，默认是下载的YOLOv8预训练权重）
            model = YOLO(self.model_path)  # 可以替换为具体的模型权重文件路径
            # 打开图片并转换为合适的格式
            img = Image.open(image_path)
            img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV格式
            # 进行预测
            results = model(img_cv2)  # 传入OpenCV格式的图片
            # 获取预测结果
            predictions = results[0].boxes  # 获取预测的框、标签等信息
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 将预测结果绘制到图片上
            img_with_predictions = img_cv2.copy()
            for box in predictions.xyxy:  # 遍历每个预测框
                x1, y1, x2, y2 = box[:4]
                confidence = box[4]  # 获取置信度
                class_probs = box[5:]  # 获取类别概率
                label = int(np.argmax(class_probs))  # 获取类别ID（最大概率）
                # 绘制边界框，类别和置信度
                cv2.rectangle(img_with_predictions, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_with_predictions, f'{model.names[label]} {confidence:.2f}', 
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 保存带有预测框的图片到 result 文件夹
            result_image_path = os.path.join(result_folder, file_name)
            cv2.imwrite(result_image_path, img_with_predictions)  # 使用OpenCV保存图片
            print(f"图片 {image_path} 预测完成，结果保存在 {result_image_path}")
            return img_with_predictions
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def predict_event(self):
        """
        跳转到预测事件处理函数，并刷新内容
        """
        # 清空当前的内容区域
        if  not self.logged_in:
            messagebox.showerror("Error", "请登录后使用功能")
        else:  
            for widget in self.main.winfo_children():
                widget.destroy()
            self.after(500,self.predict_page())   
    def predict_page(self):
        """
        加载数据处理内容的具体函数
        """
        # 数据预处理页面
        self.total_images = 0
        self.current_image_index = 0
        self.processed_images = []  # 用于存储处理后的图像
        self.model = None 
        # 按钮框架
        self.button_frame = ctk.CTkFrame(self.main, fg_color="#f5f5f5")  # 按钮框架保持深色背景
        self.button_frame.grid(row=4, column=0, padx=20, pady=10, sticky='nsew')
        # 配置主窗口和按钮框架的列
        self.main.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_columnconfigure(3, weight=1)
        self.button_frame.grid_columnconfigure(4, weight=1)
        # 按钮的样式
        button_params = {
            "width": 130,
            "height": 45,
            "corner_radius": 10,  # 更圆的角
            "font": ctk.CTkFont(size=16, weight="bold"),  # 增大字体并加粗
        }
        # 按钮颜色的自定义（现代化的深色背景）
        button_colors = {
            "green": "#5CBB5A",  # 绿色 (较浅的绿色)
            "orange": "#FF5722",  # 橙色 (鲜艳的橙色)
            "red": "#D32F2F",  # 红色 (深红色)
            "light_blue": "#1976D2",  # 蓝色 (现代蓝色)
        }
        # 选择图像文件夹按钮 (绿色)
        self.datapre_button_1 = ctk.CTkButton(
            self.button_frame, 
            text="选择图像文件夹", 
            command=self.selectdatapre_folder, 
            fg_color=button_colors["green"],  
            hover_color="#4CAF50",  # 悬停颜色
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.datapre_button_1.grid(row=4, column=0, padx=20, pady=10)
        # 选择模型文件夹按钮 (绿色)
        self.datapre_button_2 = ctk.CTkButton(
            self.button_frame, 
            text="选择模型文件夹", 
            command=self.selectpremodel_folder, 
            fg_color=button_colors["green"],  
            hover_color="#4CAF50",  
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.datapre_button_2.grid(row=4, column=1, padx=20, pady=10)
        # 开始处理按钮 (橙色)
        self.datapre_button_3 = ctk.CTkButton(
            self.button_frame, 
            text="开始识别", 
            command=self.start_processing, 
            fg_color=button_colors["orange"],  # 按钮背景颜色
            hover_color="#F57C00",  # 悬停时按钮的背景颜色
            border_color="#E65100",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_3.grid(row=4, column=2, padx=20, pady=10)
        # 上一个按钮 (红色)
        self.datapre_button_4 = ctk.CTkButton(
            self.button_frame, 
            text="上一张", 
            command=self.showlastimage, 
            fg_color=button_colors["red"],  # 按钮背景颜色
            hover_color="#C62828",  # 悬停时按钮的背景颜色
            border_color="#B71C1C",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_4.grid(row=4, column=3, padx=20, pady=10)
        # 下一个按钮 (蓝色)
        self.datapre_button_5 = ctk.CTkButton(
            self.button_frame, 
            text="下一张", 
            command=self.shownextimage, 
            fg_color=button_colors["light_blue"],  # 按钮背景颜色
            hover_color="#1565C0",  # 悬停时按钮的背景颜色
            border_color="#0288D1",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_5.grid(row=4, column=4, padx=20, pady=10)
        # 显示进度条和索引
        self.progress_bar = ctk.CTkProgressBar(self.main, progress_color="#FF9800", orientation="horizontal")
        self.progress_bar.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        # 显示图片的索引，放置在进度条旁边
        self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}", font=ctk.CTkFont(size=14))
        self.image_index_label.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        # 显示原始图片区域
        self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
        self.original_image_label.grid(row=1, column=0, padx=20, pady=20)
        # 显示处理后的图片区域
        self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
        self.processed_image_label.grid(row=2, column=0, padx=20, pady=20)
        self.model_label = ctk.CTkLabel(self.main, 
                                text="尚未选择模型", 
                                font=ctk.CTkFont(family="Segoe UI", size=16, weight="normal"))
        self.model_label.grid(row=0, column=0, padx=5, pady=5)
    def histogram_equalization(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path).convert('L')  # 转为灰度图
            # 转为numpy数组
            img_array = np.array(img)
            # 应用直方图均衡化
            equalized_img_array = cv2.equalizeHist(img_array)
            equalized_img = Image.fromarray(equalized_img_array)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存均衡化后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            equalized_img.save(result_image_path)
            print(f"图片 {image_path} 应用了直方图均衡化，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def resize_image(self, image_path, target_size=(400, 400)):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 调整图片大小
            resized_img = img.resize(target_size)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存缩放后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            resized_img.save(result_image_path)
            print(f"图片 {image_path} 被缩放到 {target_size}，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def threshold_image(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path).convert('L')  # 转为灰度图
            # 将图像转换为numpy数组
            img_array = np.array(img)
            # 应用阈值
            _, thresholded_img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
            thresholded_img = Image.fromarray(thresholded_img_array)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存二值化后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            thresholded_img.save(result_image_path)
            print(f"图片 {image_path} 应用了二值化处理，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def apply_gaussian_blur(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 应用高斯模糊
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=5))
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存模糊后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            blurred_img.save(result_image_path)
            print(f"图片 {image_path} 应用了高斯模糊，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def convert_image_to_grayscale(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 将图像转换为灰度图
            grayscale_img = img.convert('L')
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存灰度图到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            grayscale_img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 转换为灰度图，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def flip_image_horizontally(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 水平翻转
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存翻转后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            flipped_img.save(result_image_path)
            print(f"图片 {image_path} 水平翻转，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def flip_image_vertically(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 垂直翻转
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存翻转后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            flipped_img.save(result_image_path)
            print(f"图片 {image_path} 垂直翻转，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def crop_image(self, image_path, crop_box=(50, 50, 200, 200)):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 裁剪图片
            cropped_img = img.crop(crop_box)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存裁剪后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            cropped_img.save(result_image_path)
            print(f"图片 {image_path} 被裁剪，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def adjust_brightness(self, image_path, factor=1.5):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 调整亮度
            enhancer = ImageEnhance.Brightness(img)
            enhanced_img = enhancer.enhance(factor)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存调整后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            enhanced_img.save(result_image_path)
            print(f"图片 {image_path} 亮度调整为 {factor}，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def adjust_contrast(self, image_path, factor=2.0):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 调整对比度
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(factor)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存调整后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            enhanced_img.save(result_image_path)
            print(f"图片 {image_path} 对比度调整为 {factor}，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def sharpen_image(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 应用锐化滤镜
            sharpened_img = img.filter(ImageFilter.SHARPEN)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存锐化后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            sharpened_img.save(result_image_path)
            print(f"图片 {image_path} 被锐化，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")    
    def convert_image_to_sepia(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 转为RGB
            img = img.convert('RGB')
            # 获取图片像素
            pixels = img.load()
            # 遍历每个像素并进行色彩转换
            for i in range(img.width):
                for j in range(img.height):
                    r, g, b = img.getpixel((i, j))
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    # 限制最大值为255
                    if tr > 255:
                        tr = 255
                    if tg > 255:
                        tg = 255
                    if tb > 255:
                        tb = 255
                    pixels[i, j] = (tr, tg, tb)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存色彩转换后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            img.save(result_image_path)
            print(f"图片 {image_path} 转换为棕褐色，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def edge_detection(self, image_path):
        try:
            # 读取图像
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # 使用Canny边缘检测
            edges = cv2.Canny(img, 100, 200)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存边缘检测后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            cv2.imwrite(result_image_path, edges)
            print(f"图片 {image_path} 进行了边缘检测，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def adjust_saturation(self, image_path, factor=1.5):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 调整饱和度
            enhancer = ImageEnhance.Color(img)
            enhanced_img = enhancer.enhance(factor)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存调整后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            enhanced_img.save(result_image_path)
            print(f"图片 {image_path} 饱和度调整为 {factor}，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def add_noise_to_image(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            img_array = np.array(img)
            # 添加随机噪声
            noise = np.random.normal(0, 25, img_array.shape)  # 平均值为0，标准差为25
            noisy_img_array = np.clip(img_array + noise, 0, 255)  # 防止溢出
            noisy_img = Image.fromarray(np.uint8(noisy_img_array))
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存添加噪声后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            noisy_img.save(result_image_path)
            print(f"图片 {image_path} 添加噪声，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def convert_color_space(self, image_path, color_space=cv2.COLOR_BGR2HSV):
        try:
            # 读取图像
            img = cv2.imread(image_path)
            # 转换色彩空间
            converted_img = cv2.cvtColor(img, color_space)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存色彩空间转换后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            cv2.imwrite(result_image_path, converted_img)
            print(f"图片 {image_path} 转换色彩空间，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def invert_image_colors(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            img_array = np.array(img)
            # 反转颜色
            inverted_img_array = 255 - img_array
            inverted_img = Image.fromarray(np.uint8(inverted_img_array))
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存反转颜色后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            inverted_img.save(result_image_path)
            print(f"图片 {image_path} 颜色反转，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def add_watermark(self, image_path, watermark_text="Sample Watermark"):
        try:
            # 打开图片
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            # 加载字体
            font = ImageFont.load_default()
            # 设置水印位置（右下角）
            text_width, text_height = draw.textsize(watermark_text, font)
            position = (img.width - text_width - 10, img.height - text_height - 10)
            # 添加水印
            draw.text(position, watermark_text, font=font, fill=(255, 255, 255, 128))
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存加水印后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            img.save(result_image_path)
            print(f"图片 {image_path} 添加了水印，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def stitch_images(self, image_paths):
        try:
            # 读取所有图像
            images = [cv2.imread(image_path) for image_path in image_paths]
            # 拼接图像（按水平方向拼接）
            stitched_img = cv2.hconcat(images)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_paths[0])
            file_name = "stitched_image.jpg"
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存拼接后的图像到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            cv2.imwrite(result_image_path, stitched_img)
            print(f"图片拼接完成，保存为 {result_image_path}")
        except Exception as e:
            print(f"处理图像拼接时发生错误: {e}")
    def rotate_image_randomly(self, image_path):
        try:
            # 打开图片
            # self.total_images = 0
            img = Image.open(image_path)
            # 选择一个随机的角度，角度只能是 45, 90, 180, 270 中的一个
            random_angle = random.choice([45, 90, 180, 270])
            # 旋转图片并避免阴影 (expand=True 让图片扩大，避免产生透明区域)
            rotated_img = img.rotate(random_angle, expand=True, resample=Image.BICUBIC)
            rotated_img = rotated_img.resize((500, 500), Image.LANCZOS) 
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存旋转后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            rotated_img.save(result_image_path)  # 保存到 result 文件夹
            # print(f"当前处理的图片数: {len(self.processed_images)}")
            # 
            print(f"图片 {image_path} 旋转了 {random_angle} 度，保存为 {result_image_path}")
            return rotated_img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def binarize_image(self, image_path):
        try:
            # 打开图片
            threshold=128
            img = Image.open(image_path)
            # 将图片转换为灰度模式
            gray_img = img.convert('L')
            # 应用二值化，使用给定的阈值
            # 如果像素值大于阈值，则为白色（255），否则为黑色（0）
            binarized_img = gray_img.point(lambda p: 255 if p > threshold else 0)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存二值化后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            binarized_img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 已二值化，保存为 {result_image_path}")
            return binarized_img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def flip_image(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 随机选择翻转方式
            flip_type = random.choice(['horizontal', 'vertical', 'none'])  # 随机选择翻转方式
            # 根据 flip_type 来选择翻转方式
            if flip_type == 'horizontal':
                # 水平翻转（左右翻转）
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == 'vertical':
                # 垂直翻转（上下翻转）
                flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                # 不进行翻转
                flipped_img = img
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存翻转后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            flipped_img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 已被 {flip_type} 翻转，保存为 {result_image_path}")
            return flipped_img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def sharpen_image(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 随机选择锐化程度
            sharpness_factor = random.uniform(5,10)  # 随机选择锐化系数，1.0 表示不锐化，值越大越锐化
            # 创建锐化增强器
            enhancer = ImageEnhance.Sharpness(img)
            # 应用锐化
            sharpened_img = enhancer.enhance(sharpness_factor)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存锐化后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            sharpened_img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 已被锐化，保存为 {result_image_path}")
            return sharpened_img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")   
    def add_mosaic(self, image_path, mosaic_size=10):
        try:
            # 打开图片
            img = Image.open(image_path)
            # 获取图片的尺寸
            width, height = img.size
            # 将图片转换为 RGB 模式，以确保处理所有颜色通道
            img = img.convert('RGB')
            # 创建马赛克效果
            for i in range(0, width, mosaic_size):
                for j in range(0, height, mosaic_size):
                    # 获取当前块的范围
                    box = (i, j, i + mosaic_size, j + mosaic_size)
                    region = img.crop(box)  # 裁剪出当前块
                    # 计算当前块的平均颜色
                    avg_color = self.get_average_color(region)
                    # 使用平均颜色填充这个块
                    self.apply_mosaic(img, box, avg_color)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存加了马赛克效果后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 已添加马赛克效果，保存为 {result_image_path}")
            return img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def get_average_color(self, region):
        """
        计算图像区域的平均颜色。
        """
        pixels = list(region.getdata())
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        return (r, g, b)
    def apply_mosaic(self, img, box, avg_color):
        """
        给指定区域应用马赛克效果，使用平均颜色填充区域。
        """
        # 在指定区域内填充平均颜色
        for i in range(box[0], box[2]):
            for j in range(box[1], box[3]):
                img.putpixel((i, j), avg_color)
    def add_noise(self, image_path):
        try:
            # 打开图片
            img = Image.open(image_path)
            noise_factor=0.01
            # 转换为 RGB 模式，以确保我们处理的图像是 RGB 格式
            img = img.convert('RGB')
            # 将图像转换为 NumPy 数组，以便我们可以更方便地处理像素
            img_array = np.array(img)
            # 获取图像的高度和宽度
            height, width, channels = img_array.shape
            # 添加盐与胡椒噪声
            noisy_img_array = self.add_salt_pepper_noise(img_array, noise_factor)
            # 将噪声图像从 NumPy 数组转换回 PIL 图像
            noisy_img = Image.fromarray(noisy_img_array)
            # 获取文件所在目录和文件名
            folder_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            # 创建 result 文件夹，如果它不存在的话
            result_folder = os.path.join(folder_path, "result")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # 保存添加了噪声后的图片到 result 文件夹中
            result_image_path = os.path.join(result_folder, file_name)
            noisy_img.save(result_image_path)  # 保存到 result 文件夹
            print(f"图片 {image_path} 已添加随机噪声，保存为 {result_image_path}")
            return noisy_img
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
    def add_salt_pepper_noise(self, img_array, noise_factor):
        """
        向图像添加盐与胡椒噪声。
        :param img_array: 输入的图像数据 (NumPy 数组格式)
        :param noise_factor: 噪声的比例，0到1之间的值，值越大，噪声越多
        :return: 添加了噪声后的图像数据
        """
        output = img_array.copy()
        # 计算噪声的数量
        total_pixels = img_array.size
        num_salt = int(total_pixels * noise_factor * 0.5)  # 50% 为盐（白色）
        num_pepper = int(total_pixels * noise_factor * 0.5)  # 50% 为胡椒（黑色）
        # 添加盐噪声（白色噪声）
        salt_coords = [np.random.randint(0, i-1, num_salt) for i in img_array.shape]
        output[salt_coords[0], salt_coords[1], :] = 255  # 设置为白色
        # 添加胡椒噪声（黑色噪声）
        pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in img_array.shape]
        output[pepper_coords[0], pepper_coords[1], :] = 0  # 设置为黑色
        return output
    def selectdatapre_folder(self):
        # 使用 filedialog.askdirectory() 让用户选择文件夹
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder_path:
            self.selected_folder = folder_path
            self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.total_images = len(self.images)
            self.load_images_from_folder(folder_path)
            print(f"选择的文件夹路径是: {self.selected_folder}")
        else:
            print("没有选择文件夹")
    def load_images_from_folder(self, folder_path):
        """加载文件夹中的所有图片文件"""
        self.images = []
        # self.processed_images = []
        self.total_images = 0
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file_name)
                img = Image.open(img_path)
                self.images.append(img)
                self.total_images += 1
        if self.total_images > 0:
            self.update_image0(self.images[0])  # 显示第一张图像
    def update_image(self, original_img, processed_img):
        """更新图像显示区域"""
        original_img_tk = ImageTk.PhotoImage(original_img)
        self.original_image_label.configure(image=original_img_tk)
        self.original_image_label.image = original_img_tk
        # 显示处理后的图像
        processed_img_tk = ImageTk.PhotoImage(processed_img)
        self.processed_image_label.configure(image=processed_img_tk)
        self.processed_image_label.image = processed_img_tk
    def update_image0(self, original_img):
        """更新图像显示区域"""
        original_img_tk = ImageTk.PhotoImage(original_img)
        self.original_image_label.configure(image=original_img_tk)
        self.original_image_label.image = original_img_tk
    def animate_progress_bar(self):
        """
        让进度条来回动，确保在处理完成后才开始。
        """
        for i in range(101):
            self.progress_bar.set(i / 100)  # 更新进度条的当前值
            self.main.update_idletasks()  # 确保UI更新
            time.sleep(0)  # 延迟，模拟进度
        # 进度条从100到0的动画
        for i in range(100, -1, -1):
            self.progress_bar.set(i / 100)  # 更新进度条的当前值
            self.main.update_idletasks()  # 确保UI更新
            time.sleep(0)  
    def start_processing(self):
        if not self.selected_folder:
            print("请先选择一个文件夹！")
            return
        # 获取下拉框中选中的预处理方法
        chosen_option = self.selected_model_var.get()
        print(f"开始执行: {chosen_option} 处理")
        # 获取文件夹中的所有图片文件
        image_files = [f for f in os.listdir(self.selected_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        if not image_files:
            print("文件夹中没有图片文件")
            return
        # 根据选中的预处理方法执行相应操作
        self.processed_images = []
        self.total_images = 0
        threading.Thread(target=self.animate_progress_bar, daemon=True).start()
        for image_file in image_files:
            image_path = os.path.join(self.selected_folder, image_file)
            if chosen_option == "随机旋转":
                self.rotate_image_randomly(image_path)
                self.processed_images.append(self.rotate_image_randomly(image_path))
                self.total_images += 1
            elif chosen_option == "随机翻转":
                self.flip_image(image_path)
                self.processed_images.append(self.flip_image(image_path))
                self.total_images += 1
            elif chosen_option == "添加噪声":
                self.add_noise(image_path)
                self.processed_images.append(self.add_noise(image_path))
                self.total_images += 1
            elif chosen_option == "二值化":
                self.binarize_image(image_path)
                self.processed_images.append(self.binarize_image(image_path))
                self.total_images += 1
            elif chosen_option == "锐化图像":
                self.sharpen_image(image_path)
                self.processed_images.append(self. sharpen_image(image_path))
                self.total_images += 1
            elif chosen_option == "已选择模型":
                self.predict_with_yolo(image_path)
                self.processed_images.append(self. predict_with_yolo(image_path))
                self.total_images += 1
        print(f"当前处理的图片数: {len(self.processed_images)}")
        self.update_image(self.images[0],self.processed_images[0])# 
        # print(self.total_images)
    def update_progress_and_index(self):
        """
        更新进度条和图片索引标签
        """
        # 假设你已经更新了进度条，这里只更新标签
        self.image_index_label.configure(text=f"{self.current_image_index + 1}/{self.total_images}")
        progress = (self.current_image_index + 1) / self.total_images
    # 更新进度条
        self.progress_bar.set(progress)
    def showlastimage(self):
        """
        显示上一张图片
        """
        if len(self.processed_images)== 0 and self.current_image_index > 0:
             self.current_image_index -= 1
             self.update_image0(self.images[self.current_image_index])
        elif len(self.processed_images)>0 and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image(self.images[self.current_image_index],self.processed_images[self.current_image_index])
        self.update_progress_and_index()
    def shownextimage(self):
        """
        显示下一张图片
        """
        if len(self.processed_images)== 0 and self.current_image_index < self.total_images - 1:
            self.current_image_index += 1
            self.update_image0(self.images[self.current_image_index])
        elif len(self.processed_images)>0 and self.current_image_index < self.total_images - 1:
            self.current_image_index += 1
            self.update_image(self.images[self.current_image_index],self.processed_images[self.current_image_index]) 
        self.update_progress_and_index()   
    def datapre_event(self):
        """
        跳转到数据处理事件函数，并刷新内容
        """
        # 清空当前的内容区域
        if not self.logged_in:
            messagebox.showerror("Error", "请登录后使用功能.")
        else:  
            for widget in self.main.winfo_children():
                widget.destroy()
            self.after(500,self.datapre_page())
    def datapre_page(self):
        """
        加载数据处理内容的具体函数
        """
        # 数据预处理页面
        self.total_images=0
        self.current_image_index = 0
        self.processed_images = []  # 用于存储处理后的图像
        self.button_frame = ctk.CTkFrame(self.main)
        self.button_frame.grid(row=0, column=0, padx=20, pady=10, sticky='nsew')
        # self.main.grid_columnconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        # self.main.grid_rowconfigure(0, weight=1)
        # self.main.grid_rowconfigure(1, weight=1)
        # self.main.grid_rowconfigure(2, weight=0)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_columnconfigure(3, weight=1)
        self.button_frame.grid_columnconfigure(4, weight=1)
        button_params = {
            "width": 120,  
            "height": 40,  
            "corner_radius": 10,  
            "font": ctk.CTkFont(size=14, weight="bold"),  
        }
        # def show_preprocess_options(self):
        #     # 显示下拉框
        #     self.option_menu.configure(state="normal")  # 显示OptionMenu
        #     self.option_menu.grid(row=1, column=0, padx=10, pady=10)
        self.datapre_options = ["选择预处理方法","随机旋转", "随机翻转", "锐化图像", "添加噪声", "二值化"]
        # 默认选择
        self.datapre_options_visible = self.datapre_options[1:]
        self.selected_model_var = ctk.StringVar(value=self.datapre_options[0])
        # 选择文件夹按钮 (绿色)
        self.datapre_button_1 = ctk.CTkButton(
            self.button_frame, 
            text="选择文件夹", 
            command=self.selectdatapre_folder, 
            fg_color="#4CAF50",  
            hover_color="#45a049",  
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.datapre_button_1.grid(row=0, column=0, padx=20, pady=10)
        # 创建一个下拉菜单（OptionMenu）
        self.datapre_button_2= ctk.CTkOptionMenu(
            self.button_frame,
            variable=self.selected_model_var,
            values=self.datapre_options_visible,
            state="hidden",  # 初始时隐藏下拉框
            # command=self.on_preprocess_option_selected
        )
        self.datapre_button_2.grid(row=0, column=1, padx=20, pady=10)
        self.datapre_button_3 = ctk.CTkButton(
            self.button_frame, 
            text="开始处理", 
            command=self.start_processing, 
            fg_color="#FF9800",  # 按钮背景颜色
            hover_color="#F57C00",  # 悬停时按钮的背景颜色
            border_color="#E65100",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_3.grid(row=0, column=2, padx=20, pady=10)
        # 上一个 (红色)
        self.datapre_button_4 = ctk.CTkButton(
            self.button_frame, 
            text="⏮", 
            command=self.showlastimage, 
            fg_color="#F44336",  # 按钮背景颜色
            hover_color="#D32F2F",  # 悬停时按钮的背景颜色
            border_color="#C62828",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_4.grid(row=0, column=3, padx=20, pady=10)
        # 下一个 (橙色)
        self.datapre_button_5 = ctk.CTkButton(
            self.button_frame, 
            text="⏭", 
            command=self.shownextimage, 
            fg_color="#FF9800",  # 按钮背景颜色
            hover_color="#F57C00",  # 悬停时按钮的背景颜色
            border_color="#E65100",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.datapre_button_5.grid(row=0, column=4,padx=20, pady=10)
        # 显示进度条和索引
        self.progress_bar = ctk.CTkProgressBar(self.main)
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        # 显示图片的索引，放置在进度条旁边
        self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}")
        self.image_index_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")
                # 显示图片区域
        # self.image_label = ctk.CTkLabel(self.main)
        # self.image_label.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#E1E1E1",text="")
        self.original_image_label.grid(row=2, column=0, padx=20, pady=20)
        self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#E1E1E1",text="")
        self.processed_image_label.grid(row=3, column=0, padx=20, pady=20)
    def __init__(self):
        super().__init__()
        self.logged_in = False
        self.title("高原地区山体滑坡遥感智能识别系统")
        width, height = self.get_screen_size()
        self.geometry(f"{1600}x{1080}")
        self.grid_rowconfigure(0, weight=1)  # 第一行可以拉伸
        self.grid_columnconfigure(0, weight=0)  # 左侧栏所在列的宽度固定
        self.grid_columnconfigure(1, weight=1)  # 右侧内容区域（main1 和 main2）占用剩余空间
        # self.grid_columnconfigure(2, weight=1) 
        self.sidebar_frame = ctk.CTkFrame(self, width=140,height=1080, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        # self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="欢迎登录", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label.bind("<Button-1>", self.login)
        button_params = {
            "width": 120,  
            "height": 40,  
            "corner_radius": 10,  
            "font": ctk.CTkFont(size=14, weight="bold"),  
        }
        # 首页(灰色)
        self.sidebar_button_1 = ctk.CTkButton(
            self.sidebar_frame, 
            text="首页", 
            command=self.index_event, 
            fg_color="black",  
            hover_color="#45a049",  
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        # 数据预处理按钮 (绿色)
        self.sidebar_button_1 = ctk.CTkButton(
            self.sidebar_frame, 
            text="数据预处理", 
            command=self.datapre_event, 
            fg_color="#4CAF50",  
            hover_color="#45a049",  
            border_color="#388E3C",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)
        # 训练按钮 (蓝色)
        self.sidebar_button_2 = ctk.CTkButton(
            self.sidebar_frame, 
            text="训练", 
            command=self.train_event, 
            fg_color="#2196F3",  
            hover_color="#1976D2",  
            border_color="#1976D2", 
            text_color="white", 
            **button_params
        )
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)
        # 预测按钮 (红色)
        self.sidebar_button_3 = ctk.CTkButton(
            self.sidebar_frame, 
            text="预测", 
            command=self.predict_event, 
            fg_color="#F44336",  # 按钮背景颜色
            hover_color="#D32F2F",  # 悬停时按钮的背景颜色
            border_color="#C62828",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        # 摄像头按钮 (橙色)
        self.sidebar_button_4 = ctk.CTkButton(
            self.sidebar_frame, 
            text="摄像头", 
            command=self.sidebar_button_event, 
            fg_color="#FF9800",  # 按钮背景颜色
            hover_color="#F57C00",  # 悬停时按钮的背景颜色
            border_color="#E65100",  # 按钮边框颜色
            text_color="white",  # 文本颜色
            **button_params
        )
        self.sidebar_button_4.grid_forget()
        # self.sidebar_button_4.grid(row=5, column=0, padx=20, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)  
        self.sidebar_frame.grid_rowconfigure(1, weight=0) 
        self.sidebar_frame.grid_rowconfigure(2, weight=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=0)
        self.sidebar_frame.grid_rowconfigure(4, weight=0)  
        self.sidebar_frame.grid_rowconfigure(5, weight=0)
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.signature =ctk.CTkLabel(self.sidebar_frame, text="(landslide)", text_color="black", font=("Roboto Medium", 10))
        self.signature.grid(row=6, column=0, sticky='s', padx=5, pady=5)
        self.main= ctk.CTkFrame(self,height=1080, corner_radius=0,fg_color="#f1f1f1")
        self.main.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
if __name__ == "__main__":
    app = App()
    app.mainloop()




import customtkinter as ctk
import tkinter as tk
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.logged_in = False
        width, height = self.get_screen_size()
        self.geometry(f"{1600}x{1080}")

        # 设置背景颜色，网易云常见的浅色调
        self.configure(bg="#F2F2F2")

        # 左侧边栏
        self.sidebar_frame = ctk.CTkFrame(self, width=180, height=1080, corner_radius=20, fg_color="#FAFAFA")
        self.sidebar_frame.place(x=0, y=0, width=180, height=1080)

        # 欢迎登录标签
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="欢迎登录", font=ctk.CTkFont(size=20, weight="bold"), text_color="#333")
        self.logo_label.place(x=20, y=20)
        self.logo_label.bind("<Button-1>", self.login)

        # 按钮样式的通用配置
        button_params = {
            "width": 140,
            "height": 50,
            "corner_radius": 25,  # 圆角
            "font": ctk.CTkFont(size=16, weight="bold"),
        }

        # 首页按钮 (渐变色背景)
        self.sidebar_button_1 = ctk.CTkButton(
            self.sidebar_frame,
            text="首页",
            command=self.index_event,
            fg_color="#FF6C6C",  # 网易云常用的渐变色
            hover_color="#FF4B4B",
            border_color="#FF3D3D",
            text_color="white",
            **button_params
        )
        self.sidebar_button_1.place(x=20, y=80, width=140, height=50)

        # 数据预处理按钮 (绿色)
        self.sidebar_button_2 = ctk.CTkButton(
            self.sidebar_frame,
            text="数据预处理",
            command=self.datapre_event,
            fg_color="#00C853",  # 清新的绿色
            hover_color="#00B34A",
            border_color="#00A34A",
            text_color="white",
            **button_params
        )
        self.sidebar_button_2.place(x=20, y=150, width=140, height=50)

        # 训练按钮 (蓝色)
        self.sidebar_button_3 = ctk.CTkButton(
            self.sidebar_frame,
            text="训练",
            command=self.train_event,
            fg_color="#2196F3",
            hover_color="#1976D2",
            border_color="#1976D2",
            text_color="white",
            **button_params
        )
        self.sidebar_button_3.place(x=20, y=220, width=140, height=50)

        # 预测按钮 (橙色)
        self.sidebar_button_4 = ctk.CTkButton(
            self.sidebar_frame,
            text="预测",
            command=self.predict_event,
            fg_color="#FF9800",  # 橙色
            hover_color="#F57C00",
            border_color="#E65100",
            text_color="white",
            **button_params
        )
        self.sidebar_button_4.place(x=20, y=290, width=140, height=50)

        # 摄像头按钮 (灰色)
        self.sidebar_button_5 = ctk.CTkButton(
            self.sidebar_frame,
            text="摄像头",
            command=self.sidebar_button_event,
            fg_color="#B0BEC5",  # 清新的灰色
            hover_color="#90A4AE",
            border_color="#78909C",
            text_color="white",
            **button_params
        )
        self.sidebar_button_5.place(x=20, y=360, width=140, height=50)

        # 签名
        self.signature = ctk.CTkLabel(self.sidebar_frame, text="(landslide)", text_color="#888", font=("Roboto Medium", 10))
        self.signature.place(x=5, y=1040)

        # 主内容框
        self.main = ctk.CTkFrame(self, height=1080, corner_radius=20, fg_color="#FFFFFF")
        self.main.place(x=180, y=0, width=1460, height=1080)
        def train_page(self):
            """
            加载数据处理内容的具体函数
            """
            # 数据预处理页面的初始化
            self.default_yaml_path = ""
            self.default_model_path = ""
            self.default_epochs = 1
            self.default_imgsz = 640
            self.default_batch = 1
            self.queue = queue.Queue()

            # 绑定变量，保存yaml文件路径和模型路径
            self.yaml_path = tk.StringVar(value=self.default_yaml_path)
            self.model_path = tk.StringVar(value=self.default_model_path)

            # 创建主框架，作为容器承载所有组件
            self.main_frame = ctk.CTkFrame(self.main)  # 使用CTkFrame作为主容器
            self.main_frame.place(x=20, y=20, relwidth=0.95, relheight=0.95)  # 使用place管理位置和大小

            # 创建顶部按钮框架，容纳所有按钮
            self.button_frame = ctk.CTkFrame(self.main_frame)  # 创建一个框架来放按钮
            self.button_frame.place(x=0, y=0, relwidth=1, height=100)  # 顶部框架占据主容器的上方，宽度为100%

            # 定义按钮样式的通用配置
            button_params = {
                "width": 140,  # 按钮宽度
                "height": 50,  # 按钮高度
                "corner_radius": 12,  # 按钮的圆角半径
                "font": ctk.CTkFont(size=16, weight="bold"),  # 按钮字体，较大的字体使按钮显得更现代
            }

            # 选择yaml文件的按钮，绿色样式
            self.datapre_button_1 = ctk.CTkButton(
                self.button_frame, 
                text="选择yaml文件",  # 按钮文本
                command=self.select_yaml,  # 点击按钮后执行的命令
                fg_color="#4CAF50",  # 按钮的背景颜色
                hover_color="#45a049",  # 悬停时的背景颜色
                border_color="#388E3C",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_1.place(x=20, y=20)  # 设置按钮位置，距离左上角20像素

            # 选择预训练模型的按钮，红色样式
            self.datapre_button_2 = ctk.CTkButton(
                self.button_frame, 
                text="选择预训练模型",  # 按钮文本
                command=self.select_model,  # 点击按钮后执行的命令
                fg_color="#F44336",  # 按钮的背景颜色
                hover_color="#D32F2F",  # 悬停时的背景颜色
                border_color="#C62828",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_2.place(x=180, y=20)  # 设置按钮位置，距离左上角180像素

            # 开始训练的按钮，橙色样式
            self.datapre_button_3 = ctk.CTkButton(
                self.button_frame, 
                text="开始训练",  # 按钮文本
                command=self.start_training,  # 点击按钮后执行的命令
                fg_color="#FF9800",  # 按钮的背景颜色
                hover_color="#F57C00",  # 悬停时的背景颜色
                border_color="#E65100",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_3.place(x=340, y=20)  # 设置按钮位置，距离左上角340像素

            # 创建一个文本框用来显示日志，放在主框架下方
            self.logs = ctk.CTkTextbox(self.main_frame, width=1000, height=600, fg_color="#E1E1E1")  # 创建一个文本框
            self.logs.place(x=20, y=120, relwidth=1, relheight=0.75)  # 文本框放在下方，宽度填充父容器，75%的高度
            def predict_page(self):
                """
                加载数据处理内容的具体函数
                """
                # 数据预处理页面
                self.total_images = 0
                self.current_image_index = 0
                self.processed_images = []  # 用于存储处理后的图像
                self.model = None 

                # 背景和外部容器
                self.main.config(bg="#f0f0f0")  # 背景颜色为浅灰色
                
                # 按钮框架
                self.button_frame = ctk.CTkFrame(self.main, fg_color="#ffffff")  # 白色背景
                self.button_frame.place(x=30, y=30, relwidth=0.94)  # 使用place精确位置，relwidth使其相对宽度为90%

                # 按钮的样式
                button_params = {
                    "width": 150,
                    "height": 50,
                    "corner_radius": 12,  # 更圆的角
                    "font": ctk.CTkFont(size=16, weight="bold"),  # 增大字体并加粗
                }

                # 按钮颜色的自定义（现代化的深色背景）
                button_colors = {
                    "green": "#4CAF50",  # 明亮绿色
                    "orange": "#FF5722",  # 鲜艳的橙色
                    "red": "#D32F2F",  # 深红色
                    "light_blue": "#2196F3",  # 清新的蓝色
                }

                # 选择图像文件夹按钮 (绿色)
                self.datapre_button_1 = ctk.CTkButton(
                    self.button_frame, 
                    text="选择图像文件夹", 
                    command=self.selectdatapre_folder, 
                    fg_color=button_colors["green"],  
                    hover_color="#388E3C",  # 悬停颜色
                    border_color="#2C6B2F",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_1.place(x=30, y=10)  # 按钮位置放在frame内部的(30, 10)

                # 选择模型文件夹按钮 (绿色)
                self.datapre_button_2 = ctk.CTkButton(
                    self.button_frame, 
                    text="选择模型文件夹", 
                    command=self.selectpremodel_folder, 
                    fg_color=button_colors["green"],  
                    hover_color="#388E3C",  
                    border_color="#2C6B2F",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_2.place(x=200, y=10)  # 第二个按钮放置在左边框稍右的位置

                # 开始处理按钮 (橙色)
                self.datapre_button_3 = ctk.CTkButton(
                    self.button_frame, 
                    text="开始识别", 
                    command=self.start_processing, 
                    fg_color=button_colors["orange"],  
                    hover_color="#E64A19",  
                    border_color="#D32F2F",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_3.place(x=370, y=10)  # 第三个按钮稍微向右偏移

                # 上一个按钮 (红色)
                self.datapre_button_4 = ctk.CTkButton(
                    self.button_frame, 
                    text="上一张", 
                    command=self.showlastimage, 
                    fg_color=button_colors["red"],  
                    hover_color="#C2185B",  
                    border_color="#B71C1C",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_4.place(x=550, y=10)  # 位置根据需要调整

                # 下一个按钮 (蓝色)
                self.datapre_button_5 = ctk.CTkButton(
                    self.button_frame, 
                    text="下一张", 
                    command=self.shownextimage, 
                    fg_color=button_colors["light_blue"],  
                    hover_color="#1976D2",  
                    border_color="#0288D1",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_5.place(x=720, y=10)  # 位置根据需要调整

                # 显示进度条
                self.progress_bar = ctk.CTkProgressBar(self.main, progress_color="#FF9800", orientation="horizontal")
                self.progress_bar.place(x=30, y=150, width=740)  # 使用place设置精确位置和宽度

                # 显示图片的索引
                self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}", font=ctk.CTkFont(size=14, weight="bold"))
                self.image_index_label.place(x=30, y=120)  # 放置在进度条上方

                # 显示原始图片区域
                self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
                self.original_image_label.place(x=30, y=220)  # 设置图像位置

                # 显示处理后的图片区域
                self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
                self.processed_image_label.place(x=30, y=640)  # 设置处理后图像的位置

                # 显示模型选择标签
                self.model_label = ctk.CTkLabel(self.main, 
                                                text="尚未选择模型", 
                                                font=ctk.CTkFont(family="Segoe UI", size=16, weight="normal"))
                self.model_label.place(x=30, y=1050)  # 设置模型标签的位置

            def datapre_page(self):
                """
                加载数据处理内容的具体函数
                """
                # 数据预处理页面
                self.total_images = 0
                self.current_image_index = 0
                self.processed_images = []  # 用于存储处理后的图像
                
                # 设置按钮框架
                self.button_frame = ctk.CTkFrame(self.main, fg_color="#FAFAFA")
                self.button_frame.place(relx=0.5, rely=0.1, anchor="n", width=700, height=60)  # 将按钮框架置于页面上部
                
                # 设置按钮样式
                button_params = {
                    "width": 120,  
                    "height": 40,  
                    "corner_radius": 12,  
                    "font": ctk.CTkFont(size=14, weight="bold"),  
                }

                # 按钮颜色自定义
                button_colors = {
                    "green": "#4CAF50",  # 绿色
                    "orange": "#FF9800",  # 橙色
                    "red": "#F44336",  # 红色
                    "light_blue": "#03A9F4",  # 蓝色
                }

                # 按钮布局
                self.datapre_button_1 = ctk.CTkButton(
                    self.button_frame, 
                    text="选择文件夹", 
                    command=self.selectdatapre_folder, 
                    fg_color=button_colors["green"],  
                    hover_color="#388E3C",  
                    border_color="#2C6B2F",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_1.place(relx=0.05, rely=0.5, anchor="center")  # 按钮在框架内靠左对齐

                self.datapre_button_2 = ctk.CTkOptionMenu(
                    self.button_frame,
                    variable=self.selected_model_var,
                    values=self.datapre_options_visible,
                    state="hidden",  # 初始时隐藏下拉框
                )
                self.datapre_button_2.place(relx=0.25, rely=0.5, anchor="center")  # 下拉菜单置于按钮右侧

                self.datapre_button_3 = ctk.CTkButton(
                    self.button_frame, 
                    text="开始处理", 
                    command=self.start_processing, 
                    fg_color=button_colors["orange"],  
                    hover_color="#FF5722",  
                    border_color="#F4511E",  
                    text_color="white", 
                    **button_params
                )
                self.datapre_button_3.place(relx=0.45, rely=0.5, anchor="center")  # 中间按钮

                self.datapre_button_4 = ctk.CTkButton(
                    self.button_frame, 
                    text="⏮", 
                    command=self.showlastimage, 
                    fg_color=button_colors["red"],  
                    hover_color="#D32F2F",  
                    border_color="#C62828",  
                    text_color="white",  
                    **button_params
                )
                self.datapre_button_4.place(relx=0.65, rely=0.5, anchor="center")  # 上一个按钮右侧

                self.datapre_button_5 = ctk.CTkButton(
                    self.button_frame, 
                    text="⏭", 
                    command=self.shownextimage, 
                    fg_color=button_colors["light_blue"],  
                    hover_color="#0288D1",  
                    border_color="#0277BD",  
                    text_color="white",  
                    **button_params
                )
                self.datapre_button_5.place(relx=0.85, rely=0.5, anchor="center")  # 下一个按钮右侧

                # 设置进度条和索引
                self.progress_bar = ctk.CTkProgressBar(self.main, progress_color="#FF9800", orientation="horizontal")
                self.progress_bar.place(relx=0.5, rely=0.2, anchor="n", width=700, height=20)  # 进度条位于页面中上部

                # 显示图片的索引
                self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}")
                self.image_index_label.place(relx=0.5, rely=0.25, anchor="n")  # 索引标签位于进度条下方

                # 显示原始图片区域
                self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#ECECEC", text="")
                self.original_image_label.place(relx=0.5, rely=0.6, anchor="n", width=700, height=400)  # 原始图片区域

                # 显示处理后的图片区域
                self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#ECECEC", text="")
                self.processed_image_label.place(relx=0.5, rely=0.9, anchor="n", width=700, height=400)  # 处理后的图片区域



if __name__ == "__main__":
    app = App()
    app.mainloop()


class App(ctk.CTk):
 

    def show_login():
        register_frame.grid_forget()  # 隐藏注册框架
        login_frame.grid(row=0, column=0, padx=60, pady=20, sticky="nsew")  # 显示登录框架

    def show_register():
        login_frame.grid_forget()  # 隐藏登录框架
        register_frame.grid(row=0, column=0, padx=60, pady=20, sticky="nsew")  # 显示注册框架

    def login():
        username = entry_username.get()
        password = entry_password.get()
        # 在这里添加登录逻辑
        print(f"登录 - 用户名: {username}, 密码: {password}")

    def register():
        username = entry_register_username.get()
        password = entry_register_password.get()
        # 在这里添加注册逻辑
        print(f"注册 - 用户名: {username}, 密码: {password}")

    root = ctk.CTk()  # 创建窗口
    root.geometry("500x500")
    root.title("高原地区山体滑坡遥感智能识别系统")

    # 设置CTk的主题和样式
    ctk.set_appearance_mode("Dark")  # 设置暗色主题
    ctk.set_default_color_theme("dark-blue")  # 设置默认主题颜色为深蓝色

    # 获取屏幕宽度和高度，设置窗口居中
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 500
    window_height = 600
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # 创建登录框架
    login_frame = ctk.CTkFrame(master=root, width=400, height=300, corner_radius=15, bg_color="#2F3136")
    login_frame.grid(row=0, column=0, padx=60, pady=20, sticky="nsew")

    # 登录标题
    label_login = ctk.CTkLabel(master=login_frame, text="系统登录", font=("Arial", 24, "bold"), text_color="white")
    label_login.grid(row=0, column=0, columnspan=2, pady=20)

    # 用户头像
    label_avatar = ctk.CTkLabel(master=login_frame, text="👤", font=("Arial", 40), text_color="white")
    label_avatar.grid(row=1, column=0, columnspan=2, pady=10)

    # 用户名输入框
    entry_username = ctk.CTkEntry(master=login_frame, placeholder_text="请输入用户名", width=280, height=40, font=("Arial", 14))
    entry_username.grid(row=2, column=0, columnspan=2, pady=10)

    # 密码输入框
    entry_password = ctk.CTkEntry(master=login_frame, placeholder_text="请输入密码", show="*", width=280, height=40, font=("Arial", 14))
    entry_password.grid(row=3, column=0, columnspan=2, pady=10)

    # 登录按钮
    button_login = ctk.CTkButton(master=login_frame, text="登录", width=280, height=40, font=("Arial", 14), command=login, corner_radius=8, fg_color="#4CAF50", hover_color="#81C784")
    button_login.grid(row=4, column=0, columnspan=2, pady=10)

    # 注册按钮
    button_show_register = ctk.CTkButton(master=login_frame, text="注册", width=280, height=40, font=("Arial", 14), command=show_register, corner_radius=8, fg_color="#2196F3", hover_color="#64B5F6")
    button_show_register.grid(row=5, column=0, columnspan=2, pady=10)

    # "记住我"复选框
    checkbox_login = ctk.CTkCheckBox(master=login_frame, text="记住我", font=("Arial", 12), text_color="white")
    checkbox_login.grid(row=6, column=0, columnspan=2, pady=10)

    # 创建注册框架
    register_frame = ctk.CTkFrame(master=root, width=400, height=300, corner_radius=15, bg_color="#2F3136")
    label_register = ctk.CTkLabel(master=register_frame, text="系统注册", font=("Arial", 24, "bold"), text_color="white")
    label_register.grid(row=0, column=0, columnspan=2, pady=20)

    entry_register_username = ctk.CTkEntry(master=register_frame, placeholder_text="请输入用户名", width=280, height=40, font=("Arial", 14))
    entry_register_username.grid(row=1, column=0, columnspan=2, pady=10)

    entry_register_password = ctk.CTkEntry(master=register_frame, placeholder_text="请输入密码", show="*", width=280, height=40, font=("Arial", 14))
    entry_register_password.grid(row=2, column=0, columnspan=2, pady=10)

    # 注册按钮
    button_register = ctk.CTkButton(master=register_frame, text="注册", width=280, height=40, font=("Arial", 14), command=register, corner_radius=8, fg_color="#2196F3", hover_color="#64B5F6")
    button_register.grid(row=3, column=0, columnspan=2, pady=10)

    # 返回登录按钮
    button_show_login = ctk.CTkButton(master=register_frame, text="返回登录", width=280, height=40, font=("Arial", 14), command=show_login, corner_radius=8, fg_color="#4CAF50", hover_color="#81C784")
    button_show_login.grid(row=4, column=0, columnspan=2, pady=10)

    label_copyright = ctk.CTkLabel(master=root, text="(The Remote Sensing Intelligent Recognition System for Landslides in Plateau Regions)", font=("Arial", 10), text_color="white")
    label_copyright.grid(row=1, column=0, pady=10, sticky="s")

    # 初始化时显示登录界面
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # 启动应用程序
    root.mainloop()

    def __init__(self):
        
        super().__init__()
        self.logged_in = False
        width, height = self.get_screen_size()
        self.geometry(f"{1600}x{1080}")

        # 主界面配置
        self.configure(bg="#f1f1f1")

        # 左侧边栏
        self.sidebar_frame = ctk.CTkFrame(self, width=140, height=1080, corner_radius=20, fg_color="#2C3E50")
        self.sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

        # 欢迎登录标签
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="欢迎登录", font=ctk.CTkFont(size=20, weight="bold"), text_color="white")
        self.logo_label.pack(padx=20, pady=(20, 10))
        self.logo_label.bind("<Button-1>", self.login)

        # 按钮样式的通用配置
        button_params = {
            "width": 120,  
            "height": 40,  
            "corner_radius": 12,  
            "font": ctk.CTkFont(size=14, weight="bold"),  
            "border_width": 2,
            "border_color": "#2980B9",
        }

        # 首页按钮
        self.sidebar_button_1 = ctk.CTkButton(
            self.sidebar_frame, 
            text="首页", 
            command=self.index_event, 
            fg_color="#34495E",  
            hover_color="#45a049",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_1.pack(fill="x", padx=20, pady=10)

        # 数据预处理按钮
        self.sidebar_button_2 = ctk.CTkButton(
            self.sidebar_frame, 
            text="数据预处理", 
            command=self.datapre_event, 
            fg_color="#27AE60",  
            hover_color="#45a049",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_2.pack(fill="x", padx=20, pady=10)

        # 训练按钮
        self.sidebar_button_3 = ctk.CTkButton(
            self.sidebar_frame, 
            text="训练", 
            command=self.train_event, 
            fg_color="#2980B9",  
            hover_color="#1976D2",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_3.pack(fill="x", padx=20, pady=10)

        # 预测按钮
        self.sidebar_button_4 = ctk.CTkButton(
            self.sidebar_frame, 
            text="预测", 
            command=self.predict_event, 
            fg_color="#E74C3C",  
            hover_color="#D32F2F",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_4.pack(fill="x", padx=20, pady=10)

        # 摄像头按钮
        self.sidebar_button_5 = ctk.CTkButton(
            self.sidebar_frame, 
            text="摄像头", 
            command=self.sidebar_button_event, 
            fg_color="#F39C12",  
            hover_color="#F57C00",  
            text_color="white", 
            **button_params
        )
        self.sidebar_button_5.pack(fill="x", padx=20, pady=10)

        # 签名
        self.signature = ctk.CTkLabel(self.sidebar_frame, text="(landslide)", text_color="white", font=("Roboto Medium", 10))
        self.signature.pack(side="bottom", padx=5, pady=5)

        # 主内容区域
        self.main = ctk.CTkFrame(self, corner_radius=20, fg_color="#ECF0F1")
        self.main.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        def predict_page(self):
            """
            加载数据处理内容的具体函数
            """
            # 数据预处理页面
            self.total_images = 0
            self.current_image_index = 0
            self.processed_images = []  # 用于存储处理后的图像
            self.model = None 

            # 按钮框架
            self.button_frame = ctk.CTkFrame(self.main, fg_color="#f5f5f5")  # 按钮框架保持深色背景
            self.button_frame.pack(padx=20, pady=10, fill="x", expand=True)

            # 按钮的样式
            button_params = {
                "width": 130,
                "height": 45,
                "corner_radius": 10,  # 更圆的角
                "font": ctk.CTkFont(size=16, weight="bold"),  # 增大字体并加粗
            }

            # 按钮颜色的自定义（现代化的深色背景）
            button_colors = {
                "green": "#5CBB5A",  # 绿色 (较浅的绿色)
                "orange": "#FF5722",  # 橙色 (鲜艳的橙色)
                "red": "#D32F2F",  # 红色 (深红色)
                "light_blue": "#1976D2",  # 蓝色 (现代蓝色)
            }

            # 选择图像文件夹按钮 (绿色)
            self.datapre_button_1 = ctk.CTkButton(
                self.button_frame, 
                text="选择图像文件夹", 
                command=self.selectdatapre_folder, 
                fg_color=button_colors["green"],  
                hover_color="#4CAF50",  # 悬停颜色
                border_color="#388E3C",  
                text_color="white", 
                **button_params
            )
            self.datapre_button_1.pack(side="left", padx=10, pady=10, expand=True)

            # 选择模型文件夹按钮 (绿色)
            self.datapre_button_2 = ctk.CTkButton(
                self.button_frame, 
                text="选择模型文件夹", 
                command=self.selectpremodel_folder, 
                fg_color=button_colors["green"],  
                hover_color="#4CAF50",  
                border_color="#388E3C",  
                text_color="white", 
                **button_params
            )
            self.datapre_button_2.pack(side="left", padx=10, pady=10, expand=True)

            # 开始处理按钮 (橙色)
            self.datapre_button_3 = ctk.CTkButton(
                self.button_frame, 
                text="开始识别", 
                command=self.start_processing, 
                fg_color=button_colors["orange"],  # 按钮背景颜色
                hover_color="#F57C00",  # 悬停时按钮的背景颜色
                border_color="#E65100",  # 按钮边框颜色
                text_color="white",  # 文本颜色
                **button_params
            )
            self.datapre_button_3.pack(side="left", padx=10, pady=10, expand=True)

            # 上一个按钮 (红色)
            self.datapre_button_4 = ctk.CTkButton(
                self.button_frame, 
                text="上一张", 
                command=self.showlastimage, 
                fg_color=button_colors["red"],  # 按钮背景颜色
                hover_color="#C62828",  # 悬停时按钮的背景颜色
                border_color="#B71C1C",  # 按钮边框颜色
                text_color="white",  # 文本颜色
                **button_params
            )
            self.datapre_button_4.pack(side="left", padx=10, pady=10, expand=True)

            # 下一个按钮 (蓝色)
            self.datapre_button_5 = ctk.CTkButton(
                self.button_frame, 
                text="下一张", 
                command=self.shownextimage, 
                fg_color=button_colors["light_blue"],  # 按钮背景颜色
                hover_color="#1565C0",  # 悬停时按钮的背景颜色
                border_color="#0288D1",  # 按钮边框颜色
                text_color="white",  # 文本颜色
                **button_params
            )
            self.datapre_button_5.pack(side="left", padx=10, pady=10, expand=True)

            # 显示进度条和索引
            self.progress_bar = ctk.CTkProgressBar(self.main, progress_color="#FF9800", orientation="horizontal")
            self.progress_bar.pack(padx=20, pady=10, fill="x", expand=True)

            # 显示图片的索引，放置在进度条旁边
            self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}", font=ctk.CTkFont(size=14))
            self.image_index_label.pack(padx=10, pady=10, side="left")

            # 显示原始图片区域
            self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
            self.original_image_label.pack(padx=20, pady=20, fill="both", expand=True)

            # 显示处理后的图片区域
            self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#f5f5f5", text="", text_color="black")
            self.processed_image_label.pack(padx=20, pady=20, fill="both", expand=True)

            # 模型选择标签
            self.model_label = ctk.CTkLabel(self.main, 
                                            text="尚未选择模型", 
                                            font=ctk.CTkFont(family="Segoe UI", size=16, weight="normal"))
            self.model_label.pack(padx=5, pady=5)

        def train_page(self):
            """
            加载数据处理内容的具体函数
            """
            # 数据预处理页面的初始化
            self.default_yaml_path = ""
            self.default_model_path = ""
            self.default_epochs = 1
            self.default_imgsz = 640
            self.default_batch = 1
            self.queue = queue.Queue()
            
            # 绑定变量，保存yaml文件路径和模型路径
            self.yaml_path = tk.StringVar(value=self.default_yaml_path)
            self.model_path = tk.StringVar(value=self.default_model_path)

            # 创建主框架，作为容器承载所有组件
            self.main_frame = ctk.CTkFrame(self.main)  # 使用CTkFrame作为主容器
            self.main_frame.pack(padx=20, pady=20, fill='both', expand=True)  # 填满父容器，并设定内边距

            # 创建顶部按钮框架，容纳所有按钮
            self.button_frame = ctk.CTkFrame(self.main_frame)  # 创建一个框架来放按钮
            self.button_frame.pack(side='top', pady=20, fill='x', anchor='n')  # 按钮框架居上对齐，填充父容器宽度

            # 定义按钮样式的通用配置
            button_params = {
                "width": 140,  # 按钮宽度
                "height": 50,  # 按钮高度
                "corner_radius": 12,  # 按钮的圆角半径
                "font": ctk.CTkFont(size=16, weight="bold"),  # 按钮字体，较大的字体使按钮显得更现代
            }

            # 选择yaml文件的按钮，绿色样式
            self.datapre_button_1 = ctk.CTkButton(
                self.button_frame, 
                text="选择yaml文件",  # 按钮文本
                command=self.select_yaml,  # 点击按钮后执行的命令
                fg_color="#4CAF50",  # 按钮的背景颜色
                hover_color="#45a049",  # 悬停时的背景颜色
                border_color="#388E3C",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_1.pack(side='left', padx=10, pady=10)  # 水平排列，按钮之间有一定间距

            # 选择预训练模型的按钮，红色样式
            self.datapre_button_2 = ctk.CTkButton(
                self.button_frame, 
                text="选择预训练模型",  # 按钮文本
                command=self.select_model,  # 点击按钮后执行的命令
                fg_color="#F44336",  # 按钮的背景颜色
                hover_color="#D32F2F",  # 悬停时的背景颜色
                border_color="#C62828",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_2.pack(side='left', padx=10, pady=10)  # 水平排列，按钮之间有一定间距

            # 开始训练的按钮，橙色样式
            self.datapre_button_3 = ctk.CTkButton(
                self.button_frame, 
                text="开始训练",  # 按钮文本
                command=self.start_training,  # 点击按钮后执行的命令
                fg_color="#FF9800",  # 按钮的背景颜色
                hover_color="#F57C00",  # 悬停时的背景颜色
                border_color="#E65100",  # 按钮边框颜色
                text_color="white",  # 按钮文本颜色
                **button_params  # 继承按钮样式的配置
            )
            self.datapre_button_3.pack(side='left', padx=10, pady=10)  # 水平排列，按钮之间有一定间距

            # 创建一个文本框用来显示日志，放在主框架下方
            self.logs = ctk.CTkTextbox(self.main_frame, width=1000, height=900, fg_color="#E1E1E1")  # 创建一个文本框
            self.logs.pack(padx=20, pady=20, fill='both', expand=True)  # 填充父容器的剩余空间，并设定内外边距
        def datapre_page(self):
            """
            加载数据处理内容的具体函数
            """
            # 数据预处理页面
            self.total_images = 0
            self.current_image_index = 0
            self.processed_images = []  # 用于存储处理后的图像
            
            # 设置按钮框架，给框架添加一定的内外边距
            self.button_frame = ctk.CTkFrame(self.main, fg_color="#F5F5F5")
            self.button_frame.pack(padx=20, pady=10, fill="x")  # 按钮框架占据水平空间
            
            # 设置按钮样式
            button_params = {
                "width": 120,  
                "height": 40,  
                "corner_radius": 10,  
                "font": ctk.CTkFont(size=14, weight="bold"),  
            }

            # 按钮颜色的自定义
            button_colors = {
                "green": "#4CAF50",  # 绿色
                "orange": "#FF9800",  # 橙色
                "red": "#F44336",  # 红色
                "light_orange": "#FF5722",  # 深橙色
            }

            # 按钮列表
            self.datapre_button_1 = ctk.CTkButton(
                self.button_frame, 
                text="选择文件夹", 
                command=self.selectdatapre_folder, 
                fg_color=button_colors["green"],  
                hover_color="#45a049",  
                border_color="#388E3C",  
                text_color="white", 
                **button_params
            )
            self.datapre_button_1.pack(side="left", padx=10, pady=10, fill="y")

            # 创建一个下拉菜单（OptionMenu）
            self.datapre_button_2 = ctk.CTkOptionMenu(
                self.button_frame,
                variable=self.selected_model_var,
                values=self.datapre_options_visible,
                state="hidden",  # 初始时隐藏下拉框
            )
            self.datapre_button_2.pack(side="left", padx=10, pady=10, fill="y")

            self.datapre_button_3 = ctk.CTkButton(
                self.button_frame, 
                text="开始处理", 
                command=self.start_processing, 
                fg_color=button_colors["orange"],  
                hover_color="#F57C00",  
                border_color="#E65100",  
                text_color="white", 
                **button_params
            )
            self.datapre_button_3.pack(side="left", padx=10, pady=10, fill="y")

            self.datapre_button_4 = ctk.CTkButton(
                self.button_frame, 
                text="⏮", 
                command=self.showlastimage, 
                fg_color=button_colors["red"],  
                hover_color="#D32F2F",  
                border_color="#C62828",  
                text_color="white",  
                **button_params
            )
            self.datapre_button_4.pack(side="left", padx=10, pady=10, fill="y")

            self.datapre_button_5 = ctk.CTkButton(
                self.button_frame, 
                text="⏭", 
                command=self.shownextimage, 
                fg_color=button_colors["light_orange"],  
                hover_color="#FF5722",  
                border_color="#F4511E",  
                text_color="white",  
                **button_params
            )
            self.datapre_button_5.pack(side="left", padx=10, pady=10, fill="y")

            # 设置进度条和索引
            self.progress_bar = ctk.CTkProgressBar(self.main, progress_color="#FF9800", orientation="horizontal")
            self.progress_bar.pack(padx=20, pady=10, fill="x")

            # 显示图片的索引
            self.image_index_label = ctk.CTkLabel(self.main, text=f"{self.current_image_index + 1}/{self.total_images}")
            self.image_index_label.pack(padx=10, pady=10, side="left")

            # 显示原始图片区域
            self.original_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#E1E1E1", text="")
            self.original_image_label.pack(padx=20, pady=20, fill="both", expand=True)

            # 显示处理后的图片区域
            self.processed_image_label = ctk.CTkLabel(self.main, width=700, height=400, fg_color="#E1E1E1", text="")
            self.processed_image_label.pack(padx=20, pady=20, fill="both", expand=True)



if __name__ == "__main__":
    app = App()
    app.mainloop()

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.ops import nms


# 数据增强：常见的图像增强方法
class DataAugmentation:
    def __init__(self, size=(416, 416)):
        self.size = size

    def __call__(self, img, boxes=None):
        img = self.resize(img)
        if boxes is not None:
            boxes = self.resize_boxes(boxes, img.shape[0], img.shape[1])
        return img, boxes

    def resize(self, img):
        """Resize image to target size while maintaining aspect ratio"""
        return cv2.resize(img, self.size)

    def resize_boxes(self, boxes, h, w):
        """Rescale bounding boxes to match the resized image dimensions"""
        boxes[:, 0] *= w  # rescale x1
        boxes[:, 1] *= h  # rescale y1
        boxes[:, 2] *= w  # rescale x2
        boxes[:, 3] *= h  # rescale y2
        return boxes


# 非极大值抑制（NMS）函数
def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.4):
    """
    Perform Non-Maximum Suppression (NMS) on the predicted boxes
    :param predictions: list of predictions [batch, boxes, class, score, x1, y1, x2, y2]
    :param conf_threshold: confidence score threshold to keep
    :param nms_threshold: NMS threshold to decide if two boxes overlap too much
    :return: filtered predictions
    """
    final_predictions = []
    
    for batch in predictions:
        # Apply NMS for each image in the batch
        boxes = batch[:, 4:8]  # [x1, y1, x2, y2]
        scores = batch[:, 3]  # confidence score
        classes = batch[:, 2]  # class predictions
        
        # Perform NMS for each class
        keep = nms(boxes, scores, nms_threshold)
        
        # Filter out low confidence boxes
        final_preds = batch[keep]
        final_preds = final_preds[final_preds[:, 3] > conf_threshold]  # Keep high confidence boxes
        final_predictions.append(final_preds)
    
    return final_predictions


# 损失计算：例如，YOLO目标检测模型的损失计算
def compute_loss(pred, targets, model, lambda_coord=5, lambda_noobj=0.5):
    """
    Compute the YOLO loss function. This computes:
    1. The coordinate loss (e.g., for bounding boxes).
    2. The object loss (e.g., for object confidence).
    3. The class loss (e.g., for class predictions).
    :param pred: predicted outputs of the model
    :param targets: ground truth labels
    :param model: the model (needed for grid size and number of anchors)
    :param lambda_coord: coefficient for coordinate loss
    :param lambda_noobj: coefficient for the no object confidence loss
    :return: loss values
    """
    # Extract model parameters
    anchors = model.anchors
    grid_size = model.grid_size
    num_classes = model.num_classes
    
    # Compute losses for objectness, class, and coordinates
    obj_loss = 0
    class_loss = 0
    coord_loss = 0
    noobj_loss = 0
    
    # Calculate object loss (IoU with ground truth)
    for i in range(len(pred)):
        # Assuming pred[i] contains (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
        pred_obj = pred[i][:, :, :, :, 0]  # objectness score
        target_obj = targets[i][:, :, :, :, 0]  # ground truth objectness
        pred_coord = pred[i][:, :, :, :, 1:5]  # bounding box predictions
        target_coord = targets[i][:, :, :, :, 1:5]  # ground truth coordinates
        pred_class = pred[i][:, :, :, :, 5:]  # class scores
        target_class = targets[i][:, :, :, :, 5:]  # ground truth classes

        # Calculate loss for object (only where object is present)
        obj_loss += F.mse_loss(pred_obj[target_obj == 1], target_obj[target_obj == 1])

        # Calculate loss for no object (penalize false positives)
        noobj_loss += F.mse_loss(pred_obj[target_obj == 0], target_obj[target_obj == 0])
        
        # Coordinate loss: mean squared error for predicted vs. true coordinates
        coord_loss += F.mse_loss(pred_coord[target_obj == 1], target_coord[target_obj == 1])
        
        # Class loss: cross entropy for classification
        class_loss += F.cross_entropy(pred_class[target_obj == 1], target_class[target_obj == 1])
    
    # Apply weights to the losses
    total_loss = lambda_coord * coord_loss + lambda_noobj * noobj_loss + obj_loss + class_loss
    
    return total_loss


# 测试函数：计算IoU
def bbox_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU value
    """
    # Calculate intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area
    return iou
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt
from utils import *  # 包含数据增强、NMS、损失计算等实用工具
from models import YOLOv8  # 假设YOLOv8模型是该模块中的模型


# 配置超参数
batch_size = 16                   # 每批次大小
learning_rate = 0.001             # 学习率
epochs = 100                      # 训练的轮数
image_size = 640                  # 输入图像大小（YOLOv8常用较大的输入大小）
num_classes = 80                  # 类别数（以COCO为例）
train_dataset = './data/train.txt' # 训练集路径
val_dataset = './data/val.txt'    # 验证集路径

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
train_data = YoloDataset(train_dataset, image_size=image_size)
val_data = YoloDataset(val_dataset, image_size=image_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 初始化YOLOv8模型
model = YOLOv8(num_classes=num_classes).to(device)

# 使用预训练权重加载
def load_pretrained_weights(model, weights_path):
    # 加载预训练的模型权重
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])

# 选择损失函数（YOLOv8损失函数）
criterion = YoloLoss(num_classes=num_classes, ignore_thresh=0.5)  # 假设使用YOLOv8特有的损失函数

# 优化器设置（使用YOLOv8优化器设置，通常采用Adam或者SGD）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学习率调度器（如余弦退火等）
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 训练函数
def train_one_epoch(epoch, model, train_loader, optimizer, criterion):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    
    # 遍历每个batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每100个batch打印一次损失
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # 返回该epoch的平均损失
    return running_loss / len(train_loader)

# 验证函数
def validate(model, val_loader, criterion):
    model.eval()  # 设置模型为验证模式
    val_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算，减少内存使用
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    # 返回验证集的平均损失
    return val_loss / len(val_loader)

# 保存模型
def save_model(epoch, model, optimizer, loss, save_path='./yolov8.pth'):
    print(f"Saving model at epoch {epoch} with loss {loss:.4f}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

# 可视化训练过程（显示图片和预测结果）
def visualize_predictions(model, val_loader):
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            predictions = model(images)
            
            # 假设YOLOv8模型的输出需要经过后处理
            predictions = post_process(predictions)
            
            # 将预测结果转换为图片
            for i in range(len(images)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 255).astype(np.uint8)
                plt.imshow(img)
                plt.show()
            break  # 只显示一个batch的结果

# 开始训练
def train():
    for epoch in range(epochs):
        # 训练阶段
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)

        # 验证阶段
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch [{epoch}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 每5个epoch保存一次模型
        if epoch % 5 == 0:
            save_model(epoch, model, optimizer, train_loss)

        # 可视化预测结果
        if epoch % 10 == 0:
            visualize_predictions(model, val_loader)

        # 学习率调度器更新
        scheduler.step()

# 调用训练函数开始训练
if __name__ == '__main__':
    start_time = time.time()
    train()
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 PyTorch Training")

    # 类别数：不包括背景
    parser.add_argument("--num-classes", default=80, type=int, 
                        help="Number of classes (excluding background). For COCO, it's 80.")
    
    # 训练使用的设备，默认使用cuda（GPU）
    parser.add_argument("--device", default="cuda", help="Training device. Options: 'cuda', 'cpu'.")

    # batch_size：每次训练中样本的数量
    parser.add_argument("-b", "--batch-size", default=16, type=int, 
                        help="Batch size for training. Default is 16.")
    
    # 训练的总轮次
    parser.add_argument("--epochs", default=50, type=int, metavar="N", 
                        help="Number of total epochs to train.")
    
    # 初始学习率
    parser.add_argument('--lr', default=0.001, type=float, 
                        help='Initial learning rate. Default is 0.001.')
    
    # 打印频率：多少个batch打印一次训练信息
    parser.add_argument('--print-freq', default=10, type=int, 
                        help='Print frequency (in terms of batches).')

    # 使用混合精度训练。较旧的GPU可能不支持，可以设置为False来关闭。
    parser.add_argument("--amp", default=True, type=bool, 
                        help="Use mixed precision training with torch.cuda.amp. Default is True.")

    # 是否使用预训练模型进行初始化
    parser.add_argument("--pretrained", default=True, type=bool, 
                        help="Whether to use pre-trained weights (e.g., from YOLOv8 backbone).")

    # 数据集路径，YOLOv8通常需要指定训练和验证集的路径
    parser.add_argument("--train-data", default="./data/train", type=str, 
                        help="Path to the training dataset.")
    parser.add_argument("--val-data", default="./data/val", type=str, 
                        help="Path to the validation dataset.")

    # 是否启用数据增强
    parser.add_argument("--augmentation", default=True, type=bool, 
                        help="Whether to use data augmentation techniques (e.g., flipping, scaling).")

    # 模型保存的路径
    parser.add_argument("--save-dir", default="./yolov8_models", type=str, 
                        help="Directory to save the trained model checkpoints.")

    # 权重更新的策略（比如SGD、Adam）
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], type=str, 
                        help="Optimizer to use for training. Options: 'adam', 'sgd'. Default is 'adam'.")

    # 是否保存训练日志
    parser.add_argument("--save-log", default=True, type=bool, 
                        help="Whether to save training logs (e.g., tensorboard, .log files).")

    # GPU数量：用于多卡训练
    parser.add_argument("--num-gpus", default=1, type=int, 
                        help="Number of GPUs to use for training. Default is 1.")
    
    # 是否使用混合数据并行（DataParallel）进行多GPU训练
    parser.add_argument("--use-dp", default=False, type=bool, 
                        help="Whether to use Data Parallel (DP) for multi-GPU training. Default is False.")
    
    # 是否启用学习率调度器
    parser.add_argument("--lr-scheduler", default="cosine", choices=["step", "cosine", "linear"], type=str,
                        help="Learning rate scheduler. Options: 'step', 'cosine', 'linear'. Default is 'cosine'.")
    
    # 训练中是否随机裁剪图像
    parser.add_argument("--random-crop", default=True, type=bool, 
                        help="Whether to use random cropping during training.")
    
    # 输入图像大小，YOLOv8可能需要指定较高分辨率
    parser.add_argument("--image-size", default=640, type=int, 
                        help="Input image size. Default is 640 pixels.")

    args = parser.parse_args()

    return args
import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image
import time

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 PyTorch Inference")
    parser.add_argument("--num-classes", default=80, type=int, help="Number of classes.")
    parser.add_argument("--device", default="cuda", help="Device to run the model on. Options: 'cuda', 'cpu'.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for inference.")
    parser.add_argument("--image-size", default=640, type=int, help="Input image size.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the trained model.")
    parser.add_argument("--input", required=True, type=str, help="Path to input image or video.")
    parser.add_argument("--output", default="./output", type=str, help="Directory to save output.")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="Confidence threshold for predictions.")
    parser.add_argument("--iou-threshold", default=0.4, type=float, help="IoU threshold for NMS.")
    parser.add_argument("--show", default=False, type=bool, help="Whether to show the prediction.")
    return parser.parse_args()

def load_model(model_path, device):
    # Load a pre-trained YOLOv8 model
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path, image_size, device):
    # Load image
    img = Image.open(image_path).convert('RGB')
    # Resize image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

def postprocess_output(pred, conf_threshold=0.5, iou_threshold=0.4):
    # Filter boxes based on confidence threshold
    pred = pred[pred[..., 4] > conf_threshold]
    
    # Apply NMS (Non-Maximum Suppression)
    boxes = pred[..., :4]
    scores = pred[..., 4] * pred[..., 5:].max(dim=-1)[0]
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    
    # Return filtered boxes, labels, and scores
    return boxes[keep], scores[keep], pred[keep]

def draw_boxes(image, boxes, scores, class_ids, class_names, colors):
    # Convert image to numpy array
    image = np.array(image)
    
    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().cpu().numpy()
        label = f"{class_names[class_ids[i]]} {scores[i]:.2f}"
        color = colors[class_ids[i]]
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def save_output(output_dir, image_name, image):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)

def inference(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Define class names (You can change this according to the dataset you used for training)
    class_names = [str(i) for i in range(args.num_classes)]
    
    # Define colors for drawing boxes
    colors = np.random.randint(0, 255, size=(args.num_classes, 3), dtype=int)
    
    # Process input image
    img_tensor, img = preprocess_image(args.input, args.image_size, device)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        pred = model(img_tensor)  # Shape: [batch_size, num_boxes, 6 (xywh+confidence+class)]
        print(f"Inference time: {time.time() - start_time:.4f}s")
    
    # Post-process output
    boxes, scores, labels = postprocess_output(pred[0], conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold)
    
    # Convert tensor boxes to image coordinates
    boxes = boxes * img.size[0]  # scale boxes to the original image size
    
    # Draw bounding boxes and labels on the image
    output_image = draw_boxes(img, boxes, scores, labels.cpu().numpy(), class_names, colors)
    
    # Save or show the result
    save_output(args.output, Path(args.input).name, output_image)
    
    if args.show:
        cv2.imshow("Prediction", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    inference(args)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import train_loader, valid_loader  # 假设你已经准备了训练和验证数据集
from model import UNet

# 假设你已经定义了一个UNet模型，输入为2个图像（通道数为7）
model = UNet(7 * 2, 1)  # 输入两个图像对（每个图像有7个通道），输出一个单通道的二值变化图
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_loss_history = []
val_loss_history = []

# 训练过程
def main(args):
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X1, batch_X2, batch_Y in train_loader:
            batch_X1, batch_X2, batch_Y = batch_X1.to(device), batch_X2.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            # 拼接两个图像（在通道维度上拼接）
            batch_X = torch.cat((batch_X1, batch_X2), dim=1)  # 合并两个输入图像
            
            # 前向传播
            outputs = model(batch_X)
            
            # 计算损失
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X1, batch_X2, batch_Y in valid_loader:
                batch_X1, batch_X2, batch_Y = batch_X1.to(device), batch_X2.to(device), batch_Y.to(device)
                batch_X = torch.cat((batch_X1, batch_X2), dim=1)  # 合并两个输入图像
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), "model_save.pth")

    # 可视化训练过程
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    ax1.plot(train_loss_history, label='Train Loss')
    ax1.plot(val_loss_history, label='Validation Loss')
    ax1.set_title('Loss Over Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    plt.show()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument("--num-classes", default=1, type=int)  # 类别数；通常为1
    parser.add_argument("--device", default="cuda", help="training device")  # 默认使用GPU
    parser.add_argument("-b", "--batch-size", default=2, type=int)  # batch_size
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')  # 学习率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')  # 打印频率
    parser.add_argument("--amp", default=True, type=bool, help="Use mixed precision training")
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

import torch.nn as nn
from osgeo import gdal
import numpy as np
import torch
import cv2
from model import UNet  # 确保你的模型文件正确导入

# 加载模型
model = UNet(7 * 2, 1)  # 输入通道数为两个图像（每个图像有7个通道），因此总通道数为 7*2
model.load_state_dict(torch.load(r''))  # 使用r前缀解决路径问题
model.eval()

# 读取前后两张图像文件
image_file_1 = r''  # 图像1
image_file_2 = r''  # 图像2，假设是变化检测的后图

# 打开两幅图像
rsdataset_1 = gdal.Open(image_file_1)
rsdataset_2 = gdal.Open(image_file_2)

# 读取两幅图像的多个波段（假设每幅图像有7个波段）
image_data_1 = np.stack([rsdataset_1.GetRasterBand(i).ReadAsArray() for i in range(1, 8)], axis=0)
image_data_2 = np.stack([rsdataset_2.GetRasterBand(i).ReadAsArray() for i in range(1, 8)], axis=0)

# 合并两幅图像（拼接在通道维度上）
image_data = np.concatenate((image_data_1, image_data_2), axis=0)  # 拼接两个图像（7通道 + 7通道）

# 将图像数据转换为 NumPy 数组并调整大小
image_data = image_data.transpose(1, 2, 0)  # 转换为 [H, W, C] 格式
image_data_resized = cv2.resize(image_data, (256, 256))  # 调整图像大小

# 将调整后的数据转换回 [C, H, W] 格式并转换为 PyTorch 张量
image_data_resized = image_data_resized.transpose(2, 0, 1)  # 转回 [C, H, W]
test_images = torch.tensor(image_data_resized).float().unsqueeze(0)  # 扩展batch维度，变为 [1, C, H, W]

# 模型预测
outputs = model(test_images)

# 将预测结果二值化，认为值大于0.5为变化区域
predicted_mask = (outputs > 0.5).float().squeeze().detach().numpy()

# 将预测结果转换为 8 位掩码图像（0-255）
predicted_mask = (predicted_mask * 255).astype(np.uint8)

# 转换原始图像为 8 位三通道图像（用于显示）
original_image_1 = np.transpose(image_data_1, (1, 2, 0))  # 转换为 [H, W, C] 格式
original_image_1 = cv2.cvtColor(original_image_1, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

# 将预测的变化掩码调整到与原图相同的大小
predicted_mask_resized = cv2.resize(predicted_mask, (original_image_1.shape[1], original_image_1.shape[0]))

# 创建一个三通道的彩色掩码，红色通道显示预测掩码
colored_mask = np.zeros_like(original_image_1)
colored_mask[:, :, 2] = predicted_mask_resized   # 将掩码设置为红色通道

# 将彩色掩码叠加到原始图像上
result_image = cv2.addWeighted(original_image_1, 0.7, colored_mask, 0.3, 0)

# 显示带有预测结果的图像
cv2.imshow('Prediction on Original Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出预测的掩码
print(predicted_mask)

# 如果你想保存带有预测结果的图像，可以使用以下代码：
cv2.imwrite(r'output_with_prediction.png', result_image)  # 保存结果图像
import torch.nn as nn
from osgeo import gdal
import numpy as np
import torch
import cv2
import os
from unet import UNet

def predict_and_save(image_file1, image_file2, model, output_dir, threshold=0.8):
    # 加载和预处理前期图像和后期图像
    rsdataset1 = gdal.Open(image_file1)
    rsdataset2 = gdal.Open(image_file2)
    
    # 假设每个图像有三个波段，分别读取
    image_data1 = np.stack([rsdataset1.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0)
    image_data2 = np.stack([rsdataset2.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0)
    
    # 将前期和后期图像堆叠为一个输入
    test_images = torch.tensor(np.stack([image_data1, image_data2], axis=0)).float().unsqueeze(0)  # (1, 2, 3, H, W)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(test_images)
    
    # 将预测结果二值化
    predicted_mask = (outputs > threshold).float().squeeze().numpy()
    
    # 将预测结果转换为 8 位掩码图像（0-255）
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    
    # 转换原始图像为 8 位三通道图像
    original_image1 = np.transpose(image_data1, (1, 2, 0))  # 转换为 [H, W, C] 格式
    original_image1 = cv2.cvtColor(original_image1, cv2.COLOR_RGB2BGR)
    
    original_image2 = np.transpose(image_data2, (1, 2, 0))  # 转换为 [H, W, C] 格式
    original_image2 = cv2.cvtColor(original_image2, cv2.COLOR_RGB2BGR)
    
    # 创建一个三通道的彩色掩码，红色通道显示预测掩码
    colored_mask = np.zeros_like(original_image1)
    colored_mask[:, :, 2] = predicted_mask  # 将掩码设置为红色
    
    # 将彩色掩码叠加到原始图像上（使用前期图像）
    result_image = cv2.addWeighted(original_image1, 0.7, colored_mask, 0.3, 0)
    
    # 保存结果图像
    base_name1 = os.path.basename(image_file1)
    base_name2 = os.path.basename(image_file2)
    result_file = os.path.join(output_dir, f"predicted_{base_name1}_{base_name2}")
    cv2.imwrite(result_file, result_image)
    print(f"Saved prediction for {image_file1} and {image_file2} to {result_file}")

def batch_predict_and_save(image_dir, output_dir, model, threshold=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取前期和后期图像对，假设文件名是成对出现的
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for i in range(0, len(image_files), 2):  # 假设每对图像由两个文件组成
        image_file1 = os.path.join(image_dir, image_files[i])
        image_file2 = os.path.join(image_dir, image_files[i+1])
        
        # 执行预测并保存
        predict_and_save(image_file1, image_file2, model, output_dir, threshold)

# 初始化模型
model = UNet(6, 1)  # 输入为两个图像，所以输入通道数为 6（每个图像3个通道，共2个图像）
model.load_state_dict(torch.load(''))
model.eval()

# 图像输入目录和预测输出目录
image_dir = ''
output_dir = 'predictions2'

# 批量预测和保存
batch_predict_and_save(image_dir, output_dir, model, threshold=0.8)
import customtkinter as ctk
import time
import threading

class ImageProcessor:
    def __init__(self, main):
        self.main = main

        # 初始化进度条，没有最大最小值
        self.progress_bar = ctk.CTkProgressBar(self.main)
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # 创建一个按钮，点击后开始进度条动画
        self.start_button = ctk.CTkButton(self.main, text="Start", command=self.start_progress_bar)
        self.start_button.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

    def start_progress_bar(self):
        """
        启动进度条，来回动
        """
        # 使用线程来避免阻塞主线程
        threading.Thread(target=self.animate_progress_bar, daemon=True).start()

    def animate_progress_bar(self):
        """
        让进度条来回动
        """
        while True:
            # 从0到1的动画
            for i in range(101):
                self.progress_bar.set(i / 100)  # 更新进度条的当前值
                self.main.update_idletasks()  # 确保UI更新
                time.sleep(0.05)  # 延迟，模拟进度

            # 从1到0的动画
            for i in range(100, -1, -1):
                self.progress_bar.set(i / 100)  # 更新进度条的当前值
                self.main.update_idletasks()  # 确保UI更新
                time.sleep(0.05)  # 延迟，模拟进度

# 使用自定义tkinter窗口
if __name__ == "__main__":
    root = ctk.CTk()

    # 创建应用程序实例
    app = ImageProcessor(root)

    # 启动GUI
    root.mainloop()

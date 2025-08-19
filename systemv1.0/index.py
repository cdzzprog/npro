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
        label_copyright = ctk.CTkLabel(master=root, text="(Intelligent Landslides Detection System By Remote Sensing In Plateau Regions)", font=("微软雅黑", 10), text_color="black")
        label_copyright.place(relx=0.5, rely=0.95, anchor="center")
        # 初始化时显示登录界面
        login_frame.place(relx=0.5, rely=0.5, anchor="center")

        root.mainloop()


    def get_screen_size(self):
        return self.winfo_screenwidth(), self.winfo_screenheight()

    
      # 判断用户名是否只包含数字
    # def is_valid_username(username):
    #     return username.isdigit()

    # # 判断密码是否合法（这里只允许字母和数字，不允许特殊字符）
    # def is_valid_password(password):
    #     # 只允许字母和数字，其他特殊字符不允许
    #     return bool(re.match("^[A-Za-z0-9]*$", password))

    # # 注册函数
    # def registernew(username, password):
    #     # 判断用户名是否为空
    #     if not username.strip():
    #         return "用户名不能为空！"
        
    #     # 判断用户名是否符合要求（只能是数字）
    #     if not is_valid_username(username):
    #         return "用户名只能包含数字！"
        
    #     # 判断密码是否为空
    #     if not password.strip():
    #         return "密码不能为空！"
        
    #     # 判断密码是否合法（不能有非法字符）
    #     if not is_valid_password(password):
    #         return "密码包含非法字符，只能包含字母和数字！"
        
    #     # 判断用户名是否已被注册
    #     if username in user_data:
    #         return "该用户名已被注册！"
        
    #     # 注册成功
    #     user_data[username] = password
    #     return "注册成功！"

    # # 登录函数
    # def loginnew(username, password):
    #     if username not in user_data:
    #         return "用户名不存在！"
    #     elif user_data[username] != password:
    #         return "密码错误！"
    #     else:
    #         return "登录成功！"





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

    def train_page1(self, main):
        """
        初始化界面并加载必要的设置
        """
        self.main = main
        self.queue = queue.Queue()

        # 默认值
        self.default_yaml_path = ""
        self.default_model_path = ""
        self.default_epochs = 1
        self.default_imgsz = 640
        self.default_batch = 1

        # 绑定变量
        self.yaml_path = tk.StringVar(value=self.default_yaml_path)
        self.model_path = tk.StringVar(value=self.default_model_path)

        # 初始化界面元素
        self.setup_ui()

        def setup_ui(self):
            """ 设置界面布局 """
            # 顶部按钮面板
            self.button_frame = ctk.CTkFrame(self.main)
            self.button_frame.pack(padx=20, pady=10, fill='x', side='top')

            # 按钮参数配置
            button_params = {
                "width": 120,
                "height": 40,
                "corner_radius": 10,
                "font": ctk.CTkFont(size=14, weight="bold"),
            }

            # 选择yaml文件按钮 (绿色)
            self.datapre_button_1 = self.create_button(
                text="选择yaml文件", 
                command=self.select_yaml, 
                fg_color="#4CAF50", 
                hover_color="#45a049", 
                border_color="#388E3C", 
                **button_params
            )
            self.datapre_button_1.pack(side='left', padx=10, pady=10)

            # 开始训练按钮 (橙色)
            self.datapre_button_3 = self.create_button(
                text="开始训练", 
                command=self.start_training, 
                fg_color="#FF9800", 
                hover_color="#F57C00", 
                border_color="#E65100", 
                **button_params
            )
            self.datapre_button_3.pack(side='left', padx=10, pady=10)

            # 选择预训练模型按钮 (红色)
            self.datapre_button_2 = self.create_button(
                text="选择预训练模型", 
                command=self.select_model, 
                fg_color="#F44336", 
                hover_color="#D32F2F", 
                border_color="#C62828", 
                **button_params
            )
            self.datapre_button_2.pack(side='left', padx=10, pady=10)

            # 日志框
            self.logs = ctk.CTkTextbox(self.main, width=1000, height=900, fg_color="#E1E1E1")
            self.logs.pack(padx=20, pady=20, fill='both', expand=True)

        def create_button(self, text, command, fg_color, hover_color, border_color, **kwargs):
            """
            创建并返回一个配置好的按钮
            """
            return ctk.CTkButton(
                self.button_frame,
                text=text,
                command=command,
                fg_color=fg_color,
                hover_color=hover_color,
                border_color=border_color,
                text_color="white",
                **kwargs
            )    





    def train_page(self):
        """
        加载数据处理内容的具体函数
        """
        # 数据预处理页面
        # self.default_yaml_path = r"路径"
        # self.default_model_path = r"路径"
        self.default_yaml_path = ""
        self.default_model_path = ""
        self.default_epochs = 1
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



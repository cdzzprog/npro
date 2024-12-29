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

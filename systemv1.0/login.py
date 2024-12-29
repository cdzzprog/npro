import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

class YourClass:
    def __init__(self):
        self.logged_in = False  # 用于跟踪登录状态

    def login(self, event):
        print("sidebar_button click")
        registered_users = {}

        # Registration function
        def register():
            username = entry_register_username.get()
            password = entry_register_password.get()

            if username in registered_users:
                messagebox.showerror("Error", "Username is already registered.")
            else:
                registered_users[username] = password
                messagebox.showinfo("Success", "User registered successfully.")
                register_frame.pack_forget()
                show_login()

        # Login function
        def login():
            username = entry_username.get()
            password = entry_password.get()

            if username in registered_users and registered_users[username] == password:
                self.logged_in = True  # 登录成功后设置状态为 True
                messagebox.showinfo("Success", "Login successful.")
                login_frame.pack_forget()  # 登录成功后关闭登录框
                show_dashboard()  # 显示其他功能页面
            else:
                messagebox.showerror("Error", "Incorrect username or password.")

        # Function to show the registration screen
        def show_register():
            login_frame.pack_forget()
            register_frame.pack(pady=20, padx=60, fill="both", expand=True)

        # Function to show the login screen
        def show_login():
            register_frame.pack_forget()
            login_frame.pack(pady=20, padx=60, fill="both", expand=True)

        # Function to show the dashboard (other functionality after login)
        def show_dashboard():
            if not self.logged_in:  # 如果用户没有登录，禁止访问功能
                messagebox.showerror("Error", "You must be logged in to access this.")
                return
            dashboard_frame.pack(pady=20, padx=60, fill="both", expand=True)

        # Initial configuration
        root = ctk.CTk()
        root.geometry("500x400")
        root.title("High-altitude Landslide Identification System")

        # Create the login frame
        login_frame = ctk.CTkFrame(master=root)
        label_login = ctk.CTkLabel(master=login_frame, text="Login System", font=("Roboto", 24))
        label_login.pack(pady=12, padx=10)
        entry_username = ctk.CTkEntry(master=login_frame, placeholder_text="Username")
        entry_username.pack(pady=12, padx=10)
        entry_password = ctk.CTkEntry(master=login_frame, placeholder_text="Password", show="*")
        entry_password.pack(pady=12, padx=10)
        button_login = ctk.CTkButton(master=login_frame, text="Login", command=login)
        button_login.pack(pady=12, padx=10)
        button_show_register = ctk.CTkButton(master=login_frame, text="Register", command=show_register)
        button_show_register.pack(pady=12, padx=10)
        checkbox_login = ctk.CTkCheckBox(master=login_frame, text="Remember Me")
        checkbox_login.pack(pady=12, padx=10)

        # Create the register frame
        register_frame = ctk.CTkFrame(master=root)
        label_register = ctk.CTkLabel(master=register_frame, text="Register System", font=("Roboto", 24))
        label_register.pack(pady=12, padx=10)
        entry_register_username = ctk.CTkEntry(master=register_frame, placeholder_text="Username")
        entry_register_username.pack(pady=12, padx=10)
        entry_register_password = ctk.CTkEntry(master=register_frame, placeholder_text="Password", show="*")
        entry_register_password.pack(pady=12, padx=10)
        button_register = ctk.CTkButton(master=register_frame, text="Register", command=register)
        button_register.pack(pady=12, padx=10)
        button_show_login = ctk.CTkButton(master=register_frame, text="Login", command=show_login)
        button_show_login.pack(pady=12, padx=10)

        # Create the dashboard (other functionalities after login)
        dashboard_frame = ctk.CTkFrame(master=root)
        label_dashboard = ctk.CTkLabel(master=dashboard_frame, text="Dashboard", font=("Roboto", 24))
        label_dashboard.pack(pady=12, padx=10)
        button_function1 = ctk.CTkButton(master=dashboard_frame, text="Function 1", command=lambda: self.function1())
        button_function1.pack(pady=12, padx=10)
        button_function2 = ctk.CTkButton(master=dashboard_frame, text="Function 2", command=lambda: self.function2())
        button_function2.pack(pady=12, padx=10)

        # Show the login frame at the start
        login_frame.pack(pady=20, padx=60, fill="both", expand=True)

        root.mainloop()

    # Example function that can only be used after login
    def function1(self):
        if not self.logged_in:
            messagebox.showerror("Error", "You must be logged in to use this feature.")
            return
        messagebox.showinfo("Function 1", "Function 1 executed successfully.")

    def function2(self):
        if not self.logged_in:
            messagebox.showerror("Error", "You must be logged in to use this feature.")
            return
        messagebox.showinfo("Function 2", "Function 2 executed successfully.")

# Create an instance of YourClass and run the login system
app = YourClass()
app.login(None)

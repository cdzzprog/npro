from PySide6.QtWidgets import QApplication,QMainWindow,QPushButton,QLabel,QVBoxLayout,QLineEdit
from PySide6.QtCore import Qt
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # btn=QPushButton('按钮',self)
        # btn.setGeometry(100,100,100,30)
        # btn.setToolTip('这是一个按钮')
        # btn.setText('按钮111')

        line=QLineEdit('输入框',self)
        line.setGeometry(100,100,100,30)
        line.setPlaceholderText("输入内容")
        mainLayout=QVBoxLayout()
        lb=QLabel('标签',self)
        lb.setText('标签改')
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mainLayout.addWidget(lb)
        self.setLayout(mainLayout)



if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
from PySide6.QtWidgets import QApplication,QMainWindow,QPushButton,QLabel,QVBoxLayout,QLineEdit
from PySide6.QtCore import Qt
from Ui_login import Ui_Form
# from Ui_conculte import Ui_Form 



class MyWindow(QMainWindow,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    



if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
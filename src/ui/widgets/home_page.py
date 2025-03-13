from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        welcome_label = QLabel("Dijital Görüntü İşleme Dersine Hoş Geldiniz")
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)
        
        info_label = QLabel("Lütfen menüden bir ödev seçiniz")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
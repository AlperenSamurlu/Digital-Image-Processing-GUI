from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt5.QtCore import Qt
from .widgets.Assignment1Page import Assignment1Page
from .widgets.home_page import HomePage
from .widgets.Assignment2Page import Assignment2Page
from PyQt5.QtWidgets import QScrollArea

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijital Görüntü İşleme")
        self.resize(800, 600)

        from PyQt5.QtWidgets import QScrollArea

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        self.setCentralWidget(scroll_area)
        layout = QVBoxLayout(central_widget)

        header = QLabel("Dijital Görüntü İşleme\nÖğrenci No: 221229042\nAd Soyad: ALPEREN SAMURLU")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        self.home_page = HomePage()
        self.assignment1_page = Assignment1Page()
        self.assignment2_page = Assignment2Page()  # YENİ EKLEDİK

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.assignment1_page)
        self.stacked_widget.addWidget(self.assignment2_page)  # YENİ EKLEDİK

        self.stacked_widget.setCurrentWidget(self.home_page)

        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()
        assignments_menu = menubar.addMenu('Ödevler')

        home_action = assignments_menu.addAction("Ana Sayfa")
        home_action.triggered.connect(self.show_home)

        assignment1_action = assignments_menu.addAction("Ödev 1: Temel İşlevselliği Oluştur")
        assignment1_action.triggered.connect(self.show_assignment1)

        assignment2_action = assignments_menu.addAction("Ödev 2: Görüntü Operasyonları ve İnterpolasyon")
        assignment2_action.triggered.connect(self.show_assignment2)

    def show_home(self):
        self.stacked_widget.setCurrentWidget(self.home_page)

    def show_assignment1(self):
        self.stacked_widget.setCurrentWidget(self.assignment1_page)

    def show_assignment2(self):
        self.stacked_widget.setCurrentWidget(self.assignment2_page)

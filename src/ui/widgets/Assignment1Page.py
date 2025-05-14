from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QHBoxLayout, QStackedWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class Assignment1Page(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image = None

    def initUI(self):
        layout = QVBoxLayout(self)
        
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)


        upload_page = QWidget()
        upload_layout = QVBoxLayout(upload_page)
        
        upload_label = QLabel("Lütfen bir görüntü yükleyin")
        upload_layout.addWidget(upload_label, alignment=Qt.AlignCenter)
        
        upload_button = QPushButton("Görüntü Yükle")
        upload_button.clicked.connect(self.load_image)
        upload_layout.addWidget(upload_button, alignment=Qt.AlignCenter)
        
        self.stacked_widget.addWidget(upload_page)

        process_page = QWidget()
        process_layout = QVBoxLayout(process_page)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        process_layout.addWidget(self.image_label)

        buttons_layout = QHBoxLayout()
        
        grayscale_btn = QPushButton("Griye Çevir")
        grayscale_btn.clicked.connect(self.convert_to_grayscale)
        buttons_layout.addWidget(grayscale_btn)
        
        blur_btn = QPushButton("Bulanıklaştır")
        blur_btn.clicked.connect(self.blur_image)
        buttons_layout.addWidget(blur_btn)
        
        edge_btn = QPushButton("Kenar Algılama")
        edge_btn.clicked.connect(self.detect_edges)
        buttons_layout.addWidget(edge_btn)
        
        process_layout.addLayout(buttons_layout)
        
        self.stacked_widget.addWidget(process_page)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", 
                                                "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)
            self.stacked_widget.setCurrentIndex(1)  

    def display_image(self, image):
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, 
                           QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, 
                           QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def convert_to_grayscale(self):
        if self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.display_image(gray_image)

    def blur_image(self):
        if self.current_image is not None:
            blurred_image = cv2.GaussianBlur(self.current_image, (5, 5), 0)
            self.display_image(blurred_image)

    def detect_edges(self):
        if self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            self.display_image(edges)
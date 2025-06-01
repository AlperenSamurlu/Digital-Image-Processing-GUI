from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QTableWidget, QTableWidgetItem, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class Assignment3Page(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        title = QLabel("\u00d6dev 3: Geli\u015fmi\u015f G\u00f6r\u00fcnt\u00fc \u0130\u015fleme")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.image_label = QLabel("G\u00f6rsel y\u00fcklenmedi")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        upload_btn = QPushButton("G\u00f6rsel Y\u00fckle")
        upload_btn.clicked.connect(self.load_image)
        layout.addWidget(upload_btn)

        sigmoid_layout = QHBoxLayout()
        sigmoid_layout.addWidget(self.create_button("a) Standart Sigmoid", self.apply_standard_sigmoid))
        sigmoid_layout.addWidget(self.create_button("b) Kayd\u0131r\u0131lm\u0131\u015f Sigmoid", self.apply_shifted_sigmoid))
        sigmoid_layout.addWidget(self.create_button("c) E\u011fimli Sigmoid", self.apply_steep_sigmoid))
        sigmoid_layout.addWidget(self.create_button("d) \u00d6zel Fonksiyon", self.apply_custom_sigmoid))
        layout.addLayout(sigmoid_layout)

        hough_layout = QHBoxLayout()
        hough_layout.addWidget(self.create_button("Yol \u00c7izgisi Tespiti", self.apply_hough_lines))
        hough_layout.addWidget(self.create_button("G\u00f6z Tespiti", self.apply_hough_eyes))
        layout.addLayout(hough_layout)

        layout.addWidget(self.create_button("Deblurring (Motion Blur Gider)", self.apply_deblurring))
        layout.addWidget(self.create_button("Nesne Say\u0131m\u0131 ve \u00d6zellik \u00c7\u0131kar\u0131m\u0131", self.apply_object_counting))

        self.table = QTableWidget()
        layout.addWidget(self.table)

    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        return btn

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "G\u00f6rsel Se\u00e7", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)

    def display_image(self, image):
        if image is None:
            return
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            q_image = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        else:
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))

    def apply_standard_sigmoid(self):
        pass

    def apply_shifted_sigmoid(self):
        pass

    def apply_steep_sigmoid(self):
        pass

    def apply_custom_sigmoid(self):
        pass

    def apply_hough_lines(self):
        pass

    def apply_hough_eyes(self):
        pass

    def apply_deblurring(self):
        pass

    def apply_object_counting(self):
        pass

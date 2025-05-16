from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QComboBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np



class Assignment2Page(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.image_label = QLabel("Lütfen bir görüntü yükleyin")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        upload_button = QPushButton("Görüntü Yükle")
        upload_button.clicked.connect(self.load_image)
        layout.addWidget(upload_button)

        # Ölçek Faktörü Girişi
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("Ölçek faktörü (örn. 2)")
        scale_layout.addWidget(self.scale_input)

        # İnterpolasyon Yöntemi Seçimi
        self.interpolation_selector = QComboBox()
        self.interpolation_selector.addItems(["Nearest Neighbor", "Bilinear", "Bicubic", "Average"])
        scale_layout.addWidget(self.interpolation_selector)

        layout.addLayout(scale_layout)

        # Büyütme Butonu
        enlarge_button = QPushButton("Görüntüyü Büyüt")
        enlarge_button.clicked.connect(self.enlarge_image)
        layout.addWidget(enlarge_button)

        # Küçültme Butonu
        shrink_button = QPushButton("Görüntüyü Küçült")
        shrink_button.clicked.connect(self.shrink_image)
        layout.addWidget(shrink_button)

        # Zoom In Butonu
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.clicked.connect(self.zoom_in)
        layout.addWidget(zoom_in_button)

        # Zoom Out Butonu
        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        layout.addWidget(zoom_out_button)

        # Açı Girişi
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Döndürme açısı (örn. 45)")
        layout.addWidget(self.rotation_input)

        # Döndürme Butonu
        rotate_button = QPushButton("Döndür")
        rotate_button.clicked.connect(self.rotate_image)
        layout.addWidget(rotate_button)


        # Çıkış Butonu


    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)

    def display_image(self, image):
        height, width = image.shape[:2]

        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def enlarge_image(self):
        if self.current_image is None:
            return

        try:
            factor = float(self.scale_input.text())
        except ValueError:
            print("Geçersiz faktör")
            return

        method = self.interpolation_selector.currentText()
        result = self.basic_scale(self.current_image, factor, method)
        self.display_image(result)

    def basic_scale(self, image, factor, method):
        original_height, original_width = image.shape[:2]
        new_height, new_width = int(original_height * factor), int(original_width * factor)

        if len(image.shape) == 2:
            channels = 1
            new_image = np.zeros((new_height, new_width), dtype=np.uint8)
        else:
            channels = image.shape[2]
            new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                orig_y = y / factor
                orig_x = x / factor

                if method == "Average":
                    new_image[y, x] = self.average_interpolation(image, orig_x, orig_y, factor, channels)
                elif method == "Bicubic":
                    new_image[y, x] = self.bicubic_interpolation(image, orig_x, orig_y, channels)
                elif method == "Bilinear":
                    new_image[y, x] = self.bilinear_interpolation(image, orig_x, orig_y, channels)
                else:  # Nearest Neighbor
                    nearest_y = min(int(round(orig_y)), original_height - 1)
                    nearest_x = min(int(round(orig_x)), original_width - 1)
                    new_image[y, x] = image[nearest_y, nearest_x]

        return new_image

    def average_interpolation(self, image, x, y, factor, channels):
        start_x = int(np.floor(x))
        start_y = int(np.floor(y))
        end_x = min(start_x + int(factor), image.shape[1])
        end_y = min(start_y + int(factor), image.shape[0])

        region = image[start_y:end_y, start_x:end_x]

        if channels == 1:
            avg_value = np.mean(region)
        else:
            avg_value = np.mean(region, axis=(0, 1))

        return np.clip(avg_value, 0, 255).astype(np.uint8)

    def bilinear_interpolation(self, image, x, y, channels):
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, image.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, image.shape[0] - 1)

        a = x - x0
        b = y - y0

        if channels == 1:
            top = (1 - a) * image[y0, x0] + a * image[y0, x1]
            bottom = (1 - a) * image[y1, x0] + a * image[y1, x1]
            value = (1 - b) * top + b * bottom
        else:
            top = (1 - a) * image[y0, x0, :] + a * image[y0, x1, :]
            bottom = (1 - a) * image[y1, x0, :] + a * image[y1, x1, :]
            value = (1 - b) * top + b * bottom

        return np.clip(value, 0, 255).astype(np.uint8)

    def cubic_weight(self, t):
        a = -0.5  # Catmull-Rom spline parametresi
        t = abs(t)
        if t <= 1:
            return (a + 2) * (t ** 3) - (a + 3) * (t ** 2) + 1
        elif t < 2:
            return a * (t ** 3) - 5 * a * (t ** 2) + 8 * a * t - 4 * a
        else:
            return 0
    def bicubic_interpolation(self, image, x, y, channels):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        result = np.zeros(channels)

        for m in range(-1, 3):
            for n in range(-1, 3):
                xm = min(max(x0 + m, 0), image.shape[1] - 1)
                yn = min(max(y0 + n, 0), image.shape[0] - 1)
                weight = self.cubic_weight(m - (x - x0)) * self.cubic_weight(n - (y - y0))
                if channels == 1:
                    result += weight * image[yn, xm]
                else:
                    result += weight * image[yn, xm, :]

        return np.clip(result, 0, 255).astype(np.uint8)

    def shrink_image(self):
        if self.current_image is None:
            return

        try:
            factor = float(self.scale_input.text())
            if factor <= 0:
                print("Faktör pozitif olmalı")
                return
        except ValueError:
            print("Geçersiz faktör")
            return

        method = self.interpolation_selector.currentText()
        result = self.basic_shrink(self.current_image, factor, method)
        self.display_image(result)

    def basic_shrink(self, image, factor, method):
        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]

        original_height, original_width = image.shape[:2]
        new_height, new_width = int(original_height / factor), int(original_width / factor)

        if channels == 1:
            new_image = np.zeros((new_height, new_width), dtype=np.uint8)
        else:
            new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                orig_y = min(int(y * factor), original_height - 1)
                orig_x = min(int(x * factor), original_width - 1)
                new_image[y, x] = image[orig_y, orig_x]

        return new_image

    def zoom_in(self):
        if self.current_image is None:
            return

        try:
            factor = float(self.scale_input.text())
            if factor <= 1:
                print("Zoom In için faktör 1'den büyük olmalı")
                return
        except ValueError:
            print("Geçersiz faktör")
            return

        result = self.basic_zoom_in(self.current_image, factor)
        self.display_image(result)

    def basic_zoom_in(self, image, factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height / factor), int(width / factor)

        start_y = (height - new_height) // 2
        start_x = (width - new_width) // 2

        # Merkezden kırpma
        cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]

        # Orijinal boyuta büyütme (nearest neighbor)
        return self.basic_scale(cropped, factor, "Nearest Neighbor")

    def zoom_out(self):
        if self.current_image is None:
            return

        try:
            factor = float(self.scale_input.text())
            if factor <= 1:
                print("Zoom Out için faktör 1'den büyük olmalı")
                return
        except ValueError:
            print("Geçersiz faktör")
            return

        result = self.basic_zoom_out(self.current_image, factor)
        self.display_image(result)

    def basic_zoom_out(self, image, factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)

        # Yeni boş görüntü
        if len(image.shape) == 2:
            new_image = np.zeros((new_height, new_width), dtype=np.uint8)
        else:
            new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2

        # Orta alana yerleştirme
        new_image[start_y:start_y + height, start_x:start_x + width] = image

        return new_image

    def rotate_image(self):
        if self.current_image is None:
            return

        try:
            angle = float(self.rotation_input.text())
        except ValueError:
            print("Geçersiz açı")
            return

        result = self.basic_rotate(self.current_image, angle)
        self.display_image(result)

    def basic_rotate(self, image, angle):
        angle_rad = np.radians(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)

        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2

        # Yeni boyutları hesapla
        new_w = int(abs(w * cos_val) + abs(h * sin_val))
        new_h = int(abs(h * cos_val) + abs(w * sin_val))

        new_image = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)

        new_cx, new_cy = new_w // 2, new_h // 2

        for y in range(new_h):
            for x in range(new_w):
                xt = x - new_cx
                yt = y - new_cy

                orig_x = int(cos_val * xt + sin_val * yt + cx)
                orig_y = int(-sin_val * xt + cos_val * yt + cy)

                if 0 <= orig_x < w and 0 <= orig_y < h:
                    new_image[y, x] = image[orig_y, orig_x]

        return new_image


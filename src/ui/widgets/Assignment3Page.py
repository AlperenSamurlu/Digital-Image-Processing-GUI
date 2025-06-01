from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QTableWidget, QTableWidgetItem, QStackedWidget,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import math
import pandas as pd

class Assignment3Page(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.table_data = []  # Tablo verilerini saklamak için
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        title = QLabel("\u00d6dev 3: Geli\u015fimi\u015f G\u00f6r\u00fcnt\u00fc \u0130\u015fleme")
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
        
        # Nesne sayımı ve Excel export butonları
        object_layout = QHBoxLayout()
        object_layout.addWidget(self.create_button("Nesne Say\u0131m\u0131 ve \u00d6zellik \u00c7\u0131kar\u0131m\u0131", self.apply_object_counting))
        object_layout.addWidget(self.create_button("Excel'e Aktar", self.export_to_excel))
        layout.addLayout(object_layout)

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
        if self.current_image is None:
            return
            
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()
            
        # Normalize to [0,1] range
        normalized = gray.astype(float) / 255.0
        
        # Standard sigmoid: f(x) = 1/(1+e^(-k*(x-0.5)))
        # k=10, merkez=0.5 (orta ton)
        k = 10
        result = 1.0 / (1.0 + np.exp(-k * (normalized - 0.5)))
        
        result = (result * 255).astype(np.uint8)
        
        if len(self.current_image.shape) == 3:
            result_colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(result_colored)
        else:
            self.display_image(result)

    def apply_shifted_sigmoid(self):
        if self.current_image is None:
            return
            
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()
            
        # Normalize to [0,1] range
        normalized = gray.astype(float) / 255.0
        
        # Shifted sigmoid: merkezi 0.3'e kaydır (daha koyu)
        # f(x) = 1/(1+e^(-k*(x-shift)))
        k = 10
        shift = 0.3  # Koyu tonları vurgular
        result = 1.0 / (1.0 + np.exp(-k * (normalized - shift)))
        
        result = (result * 255).astype(np.uint8)
        
        if len(self.current_image.shape) == 3:
            result_colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(result_colored)
        else:
            self.display_image(result)

    def apply_steep_sigmoid(self):
        if self.current_image is None:
            return
            
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()
            
        # Normalize to [0,1] range
        normalized = gray.astype(float) / 255.0
        
        # Steep sigmoid: çok dik eğim (keskin geçiş)
        # f(x) = 1/(1+e^(-k*(x-0.5)))
        k = 25  # Çok yüksek k değeri
        result = 1.0 / (1.0 + np.exp(-k * (normalized - 0.5)))
        
        result = (result * 255).astype(np.uint8)
        
        if len(self.current_image.shape) == 3:
            result_colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(result_colored)
        else:
            self.display_image(result)

    def apply_custom_sigmoid(self):
        if self.current_image is None:
            return
            
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()
            
        # Normalize to [0,1] range
        normalized = gray.astype(float) / 255.0
        
        # Custom sigmoid: Çift sigmoid (S-curve enhancer)
        # Koyu ve açık bölgeleri farklı şekilde işle
        result = np.zeros_like(normalized)
        
        # İlk sigmoid: koyu tonlar için (0-0.5)
        mask_dark = normalized <= 0.5
        x_dark = normalized[mask_dark] * 2  # [0,0.5] -> [0,1]
        result[mask_dark] = 0.5 * (1.0 / (1.0 + np.exp(-15 * (x_dark - 0.3))))
        
        # İkinci sigmoid: açık tonlar için (0.5-1)
        mask_bright = normalized > 0.5
        x_bright = (normalized[mask_bright] - 0.5) * 2  # [0.5,1] -> [0,1]
        result[mask_bright] = 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-12 * (x_bright - 0.7))))
        
        result = (result * 255).astype(np.uint8)
        
        if len(self.current_image.shape) == 3:
            result_colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.display_image(result_colored)
        else:
            self.display_image(result)

    def apply_hough_lines(self):
        if self.current_image is None:
            return

        image = self.current_image.copy()
        
        # HSV'ye çevir - renk segmentasyonu için
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Asfalt (koyu gri/siyah) için maske
        lower_asphalt = np.array([0, 0, 0])
        upper_asphalt = np.array([180, 255, 80])
        asphalt_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)
        
        # Kum/toprak (sarımsı/kahverengi) için maske
        lower_sand = np.array([10, 30, 80])
        upper_sand = np.array([30, 255, 200])
        sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)
        
        # ROI maskesi - sadece alt yarı
        height, width = image.shape[:2]
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_vertices = np.array([
            [0, int(height * 0.5)],
            [width, int(height * 0.5)],
            [width, height],
            [0, height]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_vertices], 255)
        
        # ROI'yi maskelere uygula
        asphalt_roi = cv2.bitwise_and(asphalt_mask, roi_mask)
        sand_roi = cv2.bitwise_and(sand_mask, roi_mask)
        
        # Morfolojik işlemler ile gürültüyü temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        asphalt_clean = cv2.morphologyEx(asphalt_roi, cv2.MORPH_CLOSE, kernel)
        sand_clean = cv2.morphologyEx(sand_roi, cv2.MORPH_CLOSE, kernel)
        
        # Asfalt ve kum arasındaki sınırları bul
        asphalt_edges = cv2.Canny(asphalt_clean, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            asphalt_edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=50,
            maxLineGap=20
        )
        
        left_lines = []
        right_lines = []
        center_x = width // 2
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çizgi merkez noktası
                line_center_x = (x1 + x2) // 2
                
                # Eğim hesapla
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    angle = math.degrees(math.atan(slope))
                    
                    # Çizgi uzunluğu kontrolü
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if line_length < 40:
                        continue
                    
                    # Sol taraf çizgileri (negatif eğim ve sol yarıda)
                    if slope < -0.3 and line_center_x < center_x:
                        left_lines.append(line)
                    # Sağ taraf çizgileri (pozitif eğim ve sağ yarıda)
                    elif slope > 0.3 and line_center_x > center_x:
                        right_lines.append(line)
        
        # Tespit edilen çizgileri çiz
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
            
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Debug için maskeleri göster (isteğe bağlı)
        # Asfalt alanlarını mavi ile göster
        image[asphalt_clean > 0] = [100, 100, 255]
        
        # Sonuçları yazdır
        cv2.putText(image, f"Left edges: {len(left_lines)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, f"Right edges: {len(right_lines)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        self.display_image(image)

    def apply_hough_eyes(self):
        if self.current_image is None:
            return
        
        # Make a copy of the image
        image = self.current_image.copy()
        
        # Load the pre-trained eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if eye_cascade.empty() or face_cascade.empty():
            print("Error loading cascade classifiers")
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces first (to limit eye search area)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # For each face, detect eyes
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                radius = int(round((ew + eh)*0.25))
                # Draw circle around the eye
                cv2.circle(image, eye_center, radius, (0,255,0), 2)
        
        # If no faces found, try direct eye detection on the whole image
        if len(faces) == 0:
            eyes = eye_cascade.detectMultiScale(gray)
            for (ex,ey,ew,eh) in eyes:
                eye_center = (ex + ew//2, ey + eh//2)
                radius = int(round((ew + eh)*0.25))
                cv2.circle(image, eye_center, radius, (0,255,0), 2)
        
        self.display_image(image)

    def apply_deblurring(self):
        if self.current_image is None:
            return
        
        # Gri tonlamaya çevir
        if len(self.current_image.shape) == 3:
            img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.current_image.copy()
        
        # Motion blur kernel oluştur (15 piksel yatay bulanıklık)
        size = 15
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        
        # Test için görüntüyü blur yap
        blurred_test = cv2.filter2D(img, -1, kernel_motion_blur)
        
        # Wiener filtresi (arkadaşınızın yaklaşımı)
        img_fft = np.fft.fft2(blurred_test)
        kernel_fft = np.fft.fft2(kernel_motion_blur, s=blurred_test.shape)
        kernel_fft_conj = np.conj(kernel_fft)
        
        # Wiener filter hesaplama
        H_abs2 = np.abs(kernel_fft) ** 2
        wiener_filter = kernel_fft_conj / (H_abs2 + 0.01)  # 0.01 gürültü parametresi
        
        # Filtreyi uygula
        result_fft = img_fft * wiener_filter
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Ek iyileştirme: Unsharp masking
        gaussian_blur = cv2.GaussianBlur(result, (0, 0), 1.0)
        sharpened = cv2.addWeighted(result, 1.5, gaussian_blur, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Karşılaştırmalı görüntü oluştur
        if len(self.current_image.shape) == 3:
            # Orijinal - Blurred - Deblurred karşılaştırması
            original_gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # Üç görüntüyü yan yana koy
            h, w = original_gray.shape
            comparison = np.zeros((h, w*3), dtype=np.uint8)
            
            comparison[:, :w] = original_gray           # Orijinal
            comparison[:, w:2*w] = blurred_test         # Bulanık
            comparison[:, 2*w:] = sharpened             # Düzeltilmiş
            
            # Etiketler ekle
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(comparison, "Blurred", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(comparison, "Deblurred", (2*w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            # Renkli görüntü olarak göster
            comparison_colored = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
            self.display_image(comparison_colored)
        else:
            # Gri tonlamalı görüntü için
            h, w = img.shape
            comparison = np.zeros((h, w*3), dtype=np.uint8)
            
            comparison[:, :w] = img                     # Orijinal
            comparison[:, w:2*w] = blurred_test         # Bulanık
            comparison[:, 2*w:] = sharpened             # Düzeltilmiş
            
            # Etiketler ekle
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(comparison, "Blurred", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(comparison, "Deblurred", (2*w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            self.display_image(comparison)

    def apply_object_counting(self):
        if self.current_image is None:
            return
        
        image = self.current_image.copy()
        original = image.copy()
        
        # Göz tespiti benzeri çoklu ölçek yaklaşımı
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Yeşil kanal çıkarımı (en güçlü yeşil sinyali için)
        b, g, r = cv2.split(image)
        green_enhanced = cv2.subtract(g, cv2.addWeighted(r, 0.5, b, 0.5, 0))
        
        # HSV'de yeşil maskesi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Kombine maske
        combined = cv2.bitwise_and(green_enhanced, green_mask)
        
        # Göz tespiti benzeri HoughCircles kullanarak dairesel nesneleri bul
        circles = cv2.HoughCircles(
            combined,
            cv2.HOUGH_GRADIENT,
            dp=1,                    # Akümülatör çözünürlüğü
            minDist=15,              # Daireler arası minimum mesafe
            param1=50,               # Canny edge üst eşiği
            param2=15,               # Merkez tespit eşiği (düşük = daha hassas)
            minRadius=3,             # Minimum yarıçap
            maxRadius=25             # Maksimum yarıçap
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Her tespit edilen daireyi kontrol et
            for (x, y, r) in circles:
                # Daire sınırları kontrolü
                if (x-r >= 0 and y-r >= 0 and 
                    x+r < image.shape[1] and y+r < image.shape[0]):
                    
                    # ROI'deki renk kontrolü
                    roi = image[y-r:y+r, x-r:x+r]
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # ROI'de yeşil piksel yüzdesi
                    roi_green_mask = cv2.inRange(roi_hsv, lower_green, upper_green)
                    green_ratio = np.sum(roi_green_mask > 0) / (roi_green_mask.shape[0] * roi_green_mask.shape[1])
                    
                    # Yeşil yoğunluğu yeterli ise kabul et
                    if green_ratio > 0.1:  # %10 yeşil piksel
                        detected_circles.append((x, y, r))
        
        # Eğer HoughCircles yeterli sonuç vermezse, template matching benzeri yaklaşım
        if len(detected_circles) < 10:  # Beklenen sayıdan az ise
            # Kontur tabanlı ek tespit
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 500:  # Alan filtresi
                    # Dairesellik kontrolü
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.3:  # Dairesel şekil
                            # Merkez ve yarıçap hesapla
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            x, y, radius = int(x), int(y), int(radius)
                            
                            # Çakışma kontrolü
                            is_duplicate = False
                            for (ex_x, ex_y, ex_r) in detected_circles:
                                dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                                if dist < 15:  # Çok yakın
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate and radius > 2:
                                detected_circles.append((x, y, radius))
        
        # Sonuçları görselleştir
        result_image = original.copy()
        
        # Tabloyu hazırla
        self.table.clear()
        self.table.setRowCount(len(detected_circles))
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(['No', 'Center', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean', 'Median'])
        
        # Tablo verilerini saklamak için listeyi temizle
        self.table_data = []
        
        # Her tespit edilen daireyi analiz et
        for i, (cx, cy, radius) in enumerate(detected_circles):
            # Daireyi çiz (göz tespiti gibi)
            cv2.circle(result_image, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(result_image, (cx, cy), 2, (0, 0, 255), -1)  # Merkez nokta
            
            # Numara etiketi
            cv2.putText(result_image, f"{i+1}", (cx-10, cy-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Bounding box boyutları
            length = width = radius * 2
            diagonal = int(radius * 2 * np.sqrt(2))
            
            # ROI analizi
            x1, y1 = max(0, cx-radius), max(0, cy-radius)
            x2, y2 = min(image.shape[1], cx+radius), min(image.shape[0], cy+radius)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Daire maskesi oluştur
                mask = np.zeros(roi.shape, dtype=np.uint8)
                center_in_roi = (cx-x1, cy-y1)
                cv2.circle(mask, center_in_roi, radius, 255, -1)
                
                # Sadece daire içindeki pikseller
                roi_pixels = roi[mask > 0]
                
                if len(roi_pixels) > 0:
                    mean_val = np.mean(roi_pixels)
                    median_val = np.median(roi_pixels)
                    
                    # Energy hesaplama
                    normalized_pixels = roi_pixels.astype(float) / 255.0
                    energy = np.sum(normalized_pixels ** 2)
                    
                    # Entropy hesaplama
                    hist, _ = np.histogram(roi_pixels, bins=256, range=(0, 256))
                    hist = hist / hist.sum()
                    hist = hist[hist > 0]
                    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
                else:
                    mean_val = median_val = energy = entropy = 0
            else:
                mean_val = median_val = energy = entropy = 0
            
            # Merkez koordinatı
            center_str = f"{cx},{cy}"
            
            # Tablo verilerini sakla
            row_data = {
                'No': i+1,
                'Center': center_str,
                'Length': f"{length} px",
                'Width': f"{width} px", 
                'Diagonal': f"{diagonal} px",
                'Energy': f"{energy:.3f}",
                'Entropy': f"{entropy:.2f}",
                'Mean': f"{mean_val:.0f}",
                'Median': f"{median_val:.0f}"
            }
            self.table_data.append(row_data)
            
            # Tabloyu doldur
            self.table.setItem(i, 0, QTableWidgetItem(f"{i+1}"))
            self.table.setItem(i, 1, QTableWidgetItem(center_str))
            self.table.setItem(i, 2, QTableWidgetItem(f"{length} px"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{width} px"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{diagonal} px"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{energy:.3f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{entropy:.2f}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{mean_val:.0f}"))
            self.table.setItem(i, 8, QTableWidgetItem(f"{median_val:.0f}"))
        
        # Toplam tespit sayısı
        cv2.putText(result_image, f"Tespit Edilen Yesil Daireler: {len(detected_circles)}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        self.display_image(result_image)

    def export_to_excel(self):
        """Tablo verilerini Excel dosyasına aktar"""
        if not self.table_data:
            QMessageBox.warning(self, "Uyarı", "Önce nesne sayımı yapınız!")
            return
        
        try:
            # Dosya kaydetme dialog'u
            file_name, _ = QFileDialog.getSaveFileName(
                self, 
                "Excel Dosyası Kaydet", 
                "koyu_yesil_alanlar.xlsx", 
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_name:
                # DataFrame oluştur
                df = pd.DataFrame(self.table_data)
                
                # Excel'e yaz
                with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Koyu Yeşil Alanlar', index=False)
                    
                    # Worksheet'i al ve formatla
                    worksheet = writer.sheets['Koyu Yeşil Alanlar']
                    
                    # Sütun genişliklerini ayarla
                    column_widths = {
                        'A': 8,   # No
                        'B': 15,  # Center
                        'C': 12,  # Length
                        'D': 12,  # Width
                        'E': 12,  # Diagonal
                        'F': 12,  # Energy
                        'G': 12,  # Entropy
                        'H': 10,  # Mean
                        'I': 10   # Median
                    }
                    
                    for col, width in column_widths.items():
                        worksheet.column_dimensions[col].width = width
                
                QMessageBox.information(self, "Başarılı", f"Veriler başarıyla kaydedildi:\n{file_name}")
                
        except ImportError:
            QMessageBox.critical(self, "Hata", "Excel export için 'pandas' ve 'openpyxl' kütüphaneleri gereklidir.\n\nKurulum:\npip install pandas openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Excel dosyası kaydedilirken hata oluştu:\n{str(e)}")

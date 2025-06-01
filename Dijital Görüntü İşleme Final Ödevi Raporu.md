# Dijital Görüntü İşleme Final Ödevi Raporu

**Öğrenci Numarası:** 221229042 
**Ders:** Dijital Görüntü İşleme  
**Öğretim Üyesi:** Dr. Öğr. Üyesi Burak YILMAZ  
**Ödev Başlığı:** Sigmoid Tabanlı Kontrast Artırımı, Hough Dönüşümü, Deblurring ve Nesne Sayımı

## İçindekiler

1. Giriş+
2. S-Curve Kontrast Güçlendirme (30 Puan)
3. Hough Transform Uygulamaları (30 Puan)
4. Deblurring Algoritması (10 Puan)
5. Nesne Sayımı ve Özellik Çıkarımı (40 Puan)
6. Sonuç

## Giriş

Bu projede, ödev yönergesinde belirtilen görüntü işleme tekniklerini uygulayan PyQt5 tabanlı bir arayüz geliştirilmiştir. Uygulama, 1. ödevde hazırlanan menü yapısına entegre edilmiş olup kullanıcı dostu etkileşimli tasarım prensiplerine uygun olarak geliştirilmiştir.

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\arayuz.png)

## S-Curve Kontrast Güçlendirme (30 Puan)

### S-Curve Metodunun Teorik Açıklaması (5 Puan)

S-Curve (Sigmoid Curve) metodu, görüntülerin kontrastını artırmak için kullanılan gelişmiş bir tekniktir. Bu yöntem, sigmoid fonksiyonunun matematiksel özelliklerini kullanarak:

- **Koyu tonları** daha koyu yapar
- **Açık tonları** daha açık yapar  
- **Orta tonlarda** yumuşak geçişler sağlar

Sigmoid fonksiyonun genel matematiksel formu:

```
f(x) = 1 / (1 + e^(-k*(x-shift)))
```

Bu metot özellikle:

- Biyomedikal görüntülerde hastalık tespiti
- Askeri görüntülerde casus uçak/terörist tespiti
- Güvenlik kameralarında detay artırımı

gibi kritik uygulamalarda kullanılır.

### Uygulanan Sigmoid Fonksiyonları

#### a) Standart Sigmoid Fonksiyonu (5 Puan)

- **Parametreler:** k=10, merkez=0.5
- **Özellik:** Orta tonları dengeli şekilde vurgular
- **Formül:** `f(x) = 1/(1+e^(-10*(x-0.5)))`

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\standart%20sigmoid.png)

#### b) Kaydırılmış Sigmoid Fonksiyonu (5 Puan)

- **Parametreler:** k=10, shift=0.3
- **Özellik:** Koyu tonları özellikle vurgular
- **Formül:** `f(x) = 1/(1+e^(-10*(x-0.3)))`

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\kaydirilmis%20sigmoid.png)**

#### c) Eğimli Sigmoid Fonksiyonu (5 Puan)

- **Parametreler:** k=25 (yüksek eğim)
- **Özellik:** Keskin kontrast geçişleri oluşturur
- **Formül:** `f(x) = 1/(1+e^(-25*(x-0.5)))`

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\egimli%20sigmoid.png)**

#### d) Özel Tasarlanmış Fonksiyon (10 Puan)

Çift sigmoid yaklaşımı kullanılarak geliştirilen yenilikçi algoritma:

**Algoritma Mantığı:**

1. Görüntüyü iki bölgeye ayır: Koyu tonlar (0-0.5) ve Açık tonlar (0.5-1)
2. Her bölge için farklı sigmoid parametreleri kullan
3. Sonuçları birleştir

```python
# Koyu tonlar için: [0,0.5] → [0,0.5]
if normalized <= 0.5:
    result = 0.5 * (1/(1+exp(-15*(2*x-0.3))))

# Açık tonlar için: [0.5,1] → [0.5,1]  
else:
    result = 0.5 + 0.5 * (1/(1+exp(-12*(2*x-0.7))))
```

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\ozelfonk.png)**

## Hough Transform Uygulamaları (30 Puan)

### Hough Transform Teorik Açıklaması (10 Puan)

Hough Transform, görüntülerde geometrik şekilleri tespit etmek için kullanılan güçlü bir parametre uzayı dönüşüm tekniğidir. 

**Temel Prensipler:**

- Görüntü uzayından parametre uzayına dönüşüm
- Voting mechanism ile robusluk
- Gürültüye karşı dayanıklılık

**Çizgi Tespiti için:** 

- Parametre uzayı: (ρ, θ)
- Dönüşüm: `ρ = x*cos(θ) + y*sin(θ)`

**Daire Tespiti için:**

- Parametre uzayı: (a, b, r) 
- Dönüşüm: `(x-a)² + (y-b)² = r²`

### a) Yol Çizgisi Tespiti (10 Puan)

**Algoritma Adımları:**

1. **HSV Renk Segmentasyonu:**
   
   - Asfalt maskeleme: H[0-180], S[0-255], V[0-80]
   - Kum/toprak maskeleme: H[10-30], S[30-255], V[80-200]

2. **ROI Uygulaması:**
   
   - Alt yarı bölge seçimi
   - Trapezoid maske

3. **Kenar Tespiti:**
   
   - Morfolojik temizleme
   - Canny edge detection (50, 150)

4. **Hough Line Transform:**
   
   - HoughLinesP parametreleri: rho=1, theta=π/180, threshold=30
   - Minimum çizgi uzunluğu: 50px
   - Maksimum boşluk: 20px

5. **Eğim Analizi:**
   
   - Sol çizgiler: slope < -0.3
   - Sağ çizgiler: slope > 0.3

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\yol.png)

### b) Göz Tespiti (10 Puan)

**Algoritma Yaklaşımı:**

1. **Cascade Classifier Kullanımı:**
   
   - haarcascade_frontalface_default.xml
   - haarcascade_eye.xml

2. **Hiyerarşik Tespit:**
   
   - Önce yüz tespiti (1.3 scale, 5 minNeighbors)
   - Yüz ROI'lerinde göz arama
   - Backup: Tüm görüntüde göz arama

3. **Görselleştirme:**
   
   - Yüz: Mavi dikdörtgen
   - Göz: Yeşil daire

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\goz.png)

## Deblurring Algoritması (10 Puan)

### Algoritma Akış Diyagramı

```
┌─────────────────┐
│  Giriş Görüntüsü │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Gri Dönüşüm     │
└─────────┬───────┘
          │
┌─────────▼───────┐
│Motion Blur      │
│Kernel (15x15)   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│Test Blurring    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│FFT Dönüşümü     │
└─────────┬───────┘
          │
┌─────────▼───────┐
│Wiener Filtresi  │
│H*/(|H|²+0.01)   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│Ters FFT         │
└─────────┬───────┘
          │
┌─────────▼───────┐
│Unsharp Masking  │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ Sonuç Görüntüsü │
└─────────────────┘
```

### Teknik Detaylar

**1. Motion Blur Kernel:**

```python
kernel = np.zeros((15, 15))
kernel[7, :] = 1/15  # Yatay hareket
```

**2. Wiener Filtresi:**

```python
H_conj = np.conj(H_fft)
wiener = H_conj / (|H|² + 0.01)
```

**3. Unsharp Masking:**

```python
sharpened = 1.5*result - 0.5*gaussian_blur
```

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\deblur.png)

## Nesne Sayımı ve Özellik Çıkarımı (40 Puan)

### Çoklu Tespit Algoritması

**1. Renk Segmentasyonu:**

- HSV koyu yeşil maskesi: H[25-85], S[30-255], V[30-255]
- Yeşil kanal güçlendirme

**2. HoughCircles Tespiti:**

```python
circles = cv2.HoughCircles(
    combined, cv2.HOUGH_GRADIENT,
    dp=1, minDist=15,
    param1=50, param2=15,
    minRadius=3, maxRadius=25
)
```

**3. Kontur Tabanlı Yedek Sistem:**

- Morfolojik işlemler (OPEN/CLOSE)
- Dairesellik kontrolü: `4πA/P² > 0.3`
- Çakışma kontrolü: mesafe < 15px

### Özellik Çıkarımı Formülleri

| Özellik      | Formül         | Açıklama                   |
| ------------ | -------------- | -------------------------- |
| Center       | (cx, cy)       | Daire merkez koordinatları |
| Length/Width | 2×radius       | Çap hesabı                 |
| Diagonal     | 2×radius×√2    | Köşegen uzunluğu           |
| Energy       | Σ(pixel/255)²  | Enerji hesabı              |
| Entropy      | -Σ(p×log₂(p))  | Bilgi entropisi            |
| Mean         | Σpixel/N       | Ortalama gri seviye        |
| Median       | median(pixels) | Medyan gri seviye          |

### Excel Export Sistemi

**Pandas DataFrame Kullanımı:**

```python
df = pd.DataFrame(self.table_data)
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Koyu Yeşil Alanlar')
```

![**[Ekran Görüntüsü 1: Ana arayüz entegrasyonu - Assignment3Page]**](C:\Users\asamu\OneDrive\Desktop\Klasor\3.Sınıf\Bahar\Görüntü%20İşleme\image-processing-app\ss\nesnesayimi.png)

**GitHub Repository:** `[GitHub - AlperenSamurlu/Digital-Image-Processing-GUI](https://github.com/AlperenSamurlu/Digital-Image-Processing-GUI)**(+5 bonus puan)**

---

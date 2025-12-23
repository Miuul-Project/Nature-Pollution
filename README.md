<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Educational-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/Bootcamp-Miuul-orange.svg" alt="Miuul"/>
</p>

<h1 align="center">🌍 Global CO₂ Analysis & Future Projections</h1>

<p align="center">
  <i>Sanayi Devrimi'nden günümüze küresel karbon emisyonlarının kapsamlı veri analizi ve makine öğrenmesi tabanlı gelecek projeksiyonları</i>
</p>

---

## 📖 Proje Hakkında

Bu proje, **Miuul Veri Bilimi Bootcamp** kapsamında 5 kişilik ekibimizle gerçekleştirdiğimiz kapsamlı bir iklim verisi analiz çalışmasıdır. **Our World in Data**'nın altın standart CO₂ veri setini kullanarak, basit emisyon istatistiklerinin ötesine geçip iklim değişikliğinin temel nedenlerini araştırdık.

### 🎯 Proje Hedefleri
- Küresel CO₂ emisyon trendlerini analiz etmek
- Ekonomik büyüme, nüfus dinamikleri ve enerji tüketimi arasındaki ilişkileri keşfetmek
- 2025-2028 dönemi için makine öğrenmesi tabanlı tahminler üretmek
- Stratejik politika önerileri geliştirmek

---

## 👥 Proje Ekibi

<table align="center">
  <tr>
    <th>İsim</th>
    <th>LinkedIn</th>
    <th>GitHub</th>
    <th>Website</th>
  </tr>
  <tr>
    <td><b>Alican Kaya</b></td>
    <td><a href="https://www.linkedin.com/in/alican-kaya-881650234/">LinkedIn</a></td>
    <td><a href="https://github.com/AlicanKaya192">GitHub</a></td>
    <td><a href="https://alican-kaya.com/">🌐 Portfolio</a></td>
  </tr>
  <tr>
    <td><b>Sude Şenol</b></td>
    <td><a href="https://www.linkedin.com/in/sude-%C5%9Fenol/">LinkedIn</a></td>
    <td><a href="https://github.com/sudesenoll">GitHub</a></td>
    <td>—</td>
  </tr>
  <tr>
    <td><b>Zülal Özge</b></td>
    <td><a href="https://www.linkedin.com/in/z%C3%BClal-%C3%B6zge-687488333/">LinkedIn</a></td>
    <td><a href="https://github.com/zulalozge">GitHub</a></td>
    <td>—</td>
  </tr>
  <tr>
    <td><b>Hasret Erdoğan</b></td>
    <td><a href="https://www.linkedin.com/in/hasret-erdo%C4%9Fan-5b463b278/">LinkedIn</a></td>
    <td><a href="https://github.com/hasreterdogan">GitHub</a></td>
    <td>—</td>
  </tr>
  <tr>
    <td><b>Duru Bağdadioğlu</b></td>
    <td><a href="https://www.linkedin.com/in/duru-ba%C4%9Fdadio%C4%9Flu/">LinkedIn</a></td>
    <td><a href="https://github.com/durubagdadioglu">GitHub</a></td>
    <td>—</td>
  </tr>
</table>

---

## 🔍 Analiz Modülleri

### 1️⃣ Tarihsel Trendler & Ülke Profilleri
- Küresel emisyonların yükselişini izleme
- Büyük ekonomilerin karşılaştırmalı analizi: **Çin, ABD, Hindistan, Rusya, Almanya, Türkiye**

### 2️⃣ Kirlilik Sürücüleri (Korelasyon Analizi)
- CO₂ emisyonlarının en güçlü tahmin edicilerini belirleme
- GDP, nüfus ve enerji tüketimi korelasyonları
- Isı haritaları ile karbon çıktısının "motorunu" ortaya koyma

### 3️⃣ Gelecek Projeksiyonları (ML Tahminleme)
- 2000-2024 verisi üzerinde eğitilmiş **Çok Değişkenli Polinom Regresyon** modelleri
- 2028'e kadar emisyon tahminleri
- %95 güven aralıklı projeksiyonlar

### 4️⃣ İleri Düzey Metrikler
- **Üretim vs Tüketim:** "Karbon Kaçağı" analizi
- **Karbon Yoğunluğu (CO₂/GDP):** Ekonomik büyümenin "yeşillik" ölçümü
- **Fosil Yakıt Bağımlılığı:** Kaynağa göre emisyon dağılımı (Kömür, Petrol, Doğalgaz)

---

## 🛠️ Kullanılan Teknolojiler

| Kategori | Teknolojiler |
|----------|-------------|
| **Programlama** | Python 3.8+ |
| **Veri İşleme** | Pandas, NumPy |
| **Makine Öğrenmesi** | Scikit-learn |
| **Görselleştirme** | Matplotlib, Seaborn, Folium |
| **3D Görselleştirme** | Plotly |

---

## 📂 Proje Yapısı

```
CO2-Pollution/
├── 📊 co2-data.py              # Ana analiz motoru
├── 🎨 gorsellestirme.py        # Görselleştirme fonksiyonları
├── 🌐 3D görselleştirme.py     # 3D veri görselleştirme
├── 🗺️ doga_kirliligi_haritasi.html  # İnteraktif harita
├── 📁 Datasets/
│   └── owid-co2-data.csv       # Our World in Data veri seti
├── 📁 img/                     # Oluşturulan grafikler
├── 📄 CO2_Analiz_Raporu.pdf    # Türkçe analiz raporu
├── 📄 CO2_Analysis_Report_Professional.pdf  # İngilizce rapor
├── 📄 future-earth-co2-analysis-presentation.pdf  # Sunum
└── 📋 metrics.json             # Model performans metrikleri
```

---

## 📈 Model Performansı

| Metrik | Değer |
|--------|-------|
| **RMSE** | Düşük hata oranı |
| **R² Score** | Yüksek açıklama gücü |
| **MAE** | Kabul edilebilir sapma |

> Model değerlendirmesi için `metrics.json` dosyasına bakınız.

---

## 🚀 Kurulum & Çalıştırma

```bash
# Repoyu klonlayın
git clone https://github.com/Miuul-Project/CO2-Pollution.git

# Dizine gidin
cd CO2-Pollution

# Bağımlılıkları yükleyin
pip install pandas numpy scikit-learn matplotlib seaborn folium

# Ana analizi çalıştırın
python co2-data.py

# Görselleştirmeleri oluşturun
python gorsellestirme.py
```

---

## 📊 Örnek Çıktılar

Proje çalıştırıldığında `img/` klasörüne aşağıdaki grafikler oluşturulur:
- 🌍 Küresel CO₂ trend grafikleri
- 🏳️ Ülke bazlı karşılaştırmalı analizler
- 🔥 Korelasyon ısı haritaları
- 📈 Gelecek projeksiyonları
- ⚡ Fosil yakıt karması

---

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir ve **Miuul Data Science Bootcamp** kapsamında tamamlanmıştır.

---

<p align="center">
  <b>⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐</b>
</p>

<p align="center">
  <i>Made with ❤️ by the CO₂ Analysis Team</i>
</p>

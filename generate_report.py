from fpdf import FPDF
import os
import json

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CO2 Veri Analizi Raporu', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_image(self, image_path, title=""):
        if os.path.exists(image_path):
            # Page width 210mm. Margins 20mm left, 20mm right.
            # Content width = 210 - 20 - 20 = 170mm.
            self.image(image_path, w=170)
            self.ln(2)
            if title:
                self.set_font('Arial', 'I', 10)
                self.cell(0, 5, title, 0, 1, 'C')
            self.ln(5)
        else:
            self.cell(0, 10, f"Resim bulunamadi: {image_path}", 0, 1)

pdf = PDF()
# Set equal margins: 20mm (2cm) on Left, Top, Right
pdf.set_margins(20, 20, 20)
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=20)

# Helper for Turkish chars
def tr(text):
    mapping = {
        'ğ': 'g', 'Ğ': 'G',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ç': 'c', 'Ç': 'C',
        'ö': 'o', 'Ö': 'O',
        'ü': 'u', 'Ü': 'U'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

# Content
pdf.chapter_body(tr("Bu rapor, küresel CO2 emisyon trendlerinin derinlemesine analizini sunmakta, seçilmiş ülkelerin emisyon profillerini karşılaştırmakta ve bu değişimleri yönlendiren temel faktörleri incelemektedir. Çalışma, küresel emisyon modelleri ve gelecek projeksiyonları hakkında veri odaklı içgörüler sağlamayı amaçlamaktadır."))

pdf.chapter_title(tr("Veri Hikayesi ve Değişkenler"))
pdf.chapter_body(tr("Veri Hikayesi:\n"
                    "Bu analizde kullanılan veri seti, 'Our World in Data' platformundan alınmış olup, ülkelerin tarihsel CO2 emisyonlarını ve bu emisyonları etkileyen ekonomik, demografik ve enerjik faktörleri içermektedir. "
                    "Veri seti, Sanayi Devrimi'nden günümüze kadar uzanan geniş bir zaman dilimini kapsamakta olup, küresel ısınmanın kök nedenlerini anlamak için kritik bir kaynaktır. "
                    "Biz bu çalışmada, veri kalitesini artırmak adına eksik verileri tamamladık ve analizi 1990 sonrası modern döneme odakladık.\n\n"
                    "Veri Setindeki Kategoriler (Değişkenler):\n"
                    "- co2: Toplam karbondioksit emisyonu (Milyon ton).\n"
                    "- gdp: Gayri Safi Yurtiçi Hasıla (Ekonomik büyüklük).\n"
                    "- population: Ülke nüfusu.\n"
                    "- energy_per_capita: Kişi başına düşen enerji tüketimi.\n"
                    "- co2_per_capita: Kişi başına düşen CO2 emisyonu.\n"
                    "- co2_per_gdp: Karbon yoğunluğu (Birim GSYİH başına emisyon).\n"
                    "- coal_co2, oil_co2, gas_co2: Kömür, petrol ve gaz kaynaklı emisyonlar."))

pdf.chapter_title(tr("Veri ve Metodoloji"))
pdf.chapter_body(tr("Bu analizde kullanilan yontem ve veri detaylari asagidaki gibidir:\n\n"
                    "Veri Seti: 'Our World in Data' (owid-co2-data.csv) kaynakli kuresel CO2 verileri kullanilmistir.\n\n"
                    "On Isleme (Preprocessing):\n"
                    "- Eksik Veriler: Ulke bazinda yillara gore siralanarak 'Linear Interpolation' yontemiyle doldurulmustur.\n"
                    "- Filtreleme: Analizler genelde 1990 sonrasi, tahmin modelleri ise 2000-2024 arasi verilere odaklanmistir.\n\n"
                    "Model Egitimi:\n"
                    "- Regresyon Modeli: Gelecegi tahmin etmek icin Cok Degiskenli Regresyon (Multivariate Regression) modeli kullanilmistir. Bu model, sadece zamani degil, GSYIH, Nufus, Enerji Tuketimi ve yakit turleri gibi faktorleri de hesaba katar.\n"
                    "- Egitim Seti: 2000-2024 yillari arasindaki verilerle model egitilmis, 2025-2028 icin tahmin uretilmistir.\n"
                    "- On Tahmin: Gelecek yillar icin once bagimsiz degiskenler (GSYIH vb.) tahmin edilmis, ardindan bu degerler CO2 tahmininde kullanilmistir.\n\n"
                    "Kullanilan Teknolojiler:\n"
                    "- Python: Pandas (Veri Manipulasyonu), Scikit-learn (Makine Ogrenmesi), Matplotlib & Seaborn (Gorsellestirme)."))

# Load metrics
try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    pdf.chapter_title(tr("Model Performansı"))
    pdf.chapter_body(tr(f"Modelin güvenilirliğini test etmek için veri seti {metrics.get('train_period', '2000-2018')} Eğitim ve {metrics.get('test_period', '2019-2024')} Test periyotlarına ayrılmıştır.\n"
                        f"- RMSE (Kök Ortalama Kare Hatası): {metrics['rmse']:.2f}\n"
                        f"- MAE (Ortalama Mutlak Hata): {metrics['mae']:.2f}\n"
                        f"- R² Skoru (Belirtme Katsayısı): {metrics['r2']:.2f}\n"
                        "Yüksek R² skoru ve düşük hata oranları, modelin tarihsel verileri başarıyla temsil ettiğini göstermektedir."))
except FileNotFoundError:
    print("metrics.json bulunamadi, model performansi eklenemedi.")

pdf.chapter_title(tr("1. Küresel CO2 Emisyonlarının Tarihsel Gelişimi"))
pdf.chapter_body(tr("Kuresel ortalama CO2 emisyonlari yillar icinde istikrarli bir sekilde artmaktadir."))
pdf.add_image("img/global_co2_trend.png", tr("Kuresel CO2 Trendi"))

pdf.chapter_title(tr("2. Ülke Bazlı Emisyon Profilleri ve Karşılaştırmalı Analiz"))
pdf.chapter_body(tr("Cin, ABD, Rusya, Turkiye, Almanya ve Hindistan'in CO2 emisyonlarini karsilastirdim.\n"
                    "- Cin: Son yillarda emisyonlarda buyuk bir artis goruldu.\n"
                    "- Hindistan: Hizli bir artis trendi gosteriyor, ancak kisi basi emisyonlari hala dusuk.\n"
                    "- ABD: Yuksek emisyonlara sahip ancak son zamanlarda hafif bir dusus egilimi var.\n"
                    "- Almanya ve Rusya: Nispeten istikrarli veya hafif dusus egilimi gosteriyor.\n"
                    "- Turkiye: Kademeli bir artis gosteriyor."))
pdf.add_image("img/country_co2_trend.png", tr("Ulke Bazli CO2 Trendi"))

pdf.chapter_title(tr("3. Emisyon Sürücüleri: İstatistiksel Korelasyon Analizi"))
pdf.chapter_body(tr("1990 sonrasi veriler icin CO2, GSYIH, Nufus, Kisi Basi Enerji ve Kisi Basi CO2 arasindaki iliskiyi inceledim.\n"
                    "- CO2, GSYIH ve Nufus ile yuksek korelasyona sahiptir.\n"
                    "- Kisi Basi Enerji, Kisi Basi CO2 ile guclu bir iliskiye sahiptir."))
pdf.add_image("img/correlation_matrix.png", tr("Korelasyon Matrisi"))

pdf.chapter_title(tr("4. Gelecek Projeksiyonları: Küresel CO2 Tahmini (2025-2028)"))
pdf.chapter_body(tr("2000-2024 verileriyle egitilen Polinom Regresyon modeli kullanilarak 2028'e kadar tahmin yapildi.\n"
                    "- Trend, buyuk degisiklikler olmazsa kuresel emisyonlarin artmaya devam edecegini gosteriyor."))
pdf.add_image("img/global_forecast_multivariate.png", tr("Kuresel Tahmin"))

pdf.chapter_title(tr("5. Bölgesel Tahminler ve Trend Analizi"))
pdf.chapter_body(tr("Tahmin modeli anahtar ulkelere uygulandi:\n"
                    "- Cin: Artis Egiliminde (Dik egim).\n"
                    "- Hindistan: Artis Egiliminde.\n"
                    "- Turkiye: Artis Egiliminde.\n"
                    "- ABD: Dusus Egiliminde.\n"
                    "- Almanya: Dusus Egiliminde.\n"
                    "- Rusya: Istikrarli/Hafif Artis."))
pdf.add_image("img/country_forecasts_multivariate.png", tr("Ulke Tahminleri"))

pdf.chapter_title(tr("6. Nüfus Yoğunluğu ve Kişi Başına Düşen Emisyonlar"))
pdf.chapter_body(tr("Nufusa gore emisyon yogunlugunu anlamak icin kisi basi emisyonlari inceledim.\n"
                    "- ABD en yuksek kisi basi emisyona sahip ancak dusus egiliminde.\n"
                    "- Cin'in kisi basi emisyonlari onemli olcude artti ancak hala ABD'den dusuk.\n"
                    "- Hindistan'in kisi basi emisyonlari en dusuk seviyede ancak artiyor."))
pdf.add_image("img/co2_per_capita_trend.png", tr("Kisi Basi CO2 Trendi"))

pdf.chapter_title(tr("7. Demografik Büyüme ve Emisyon İlişkisi"))
pdf.chapter_body(tr("Emisyonlarin nufustan daha hizli buyuyup buyumedigini gormek icin karsilastirma yaptim.\n\n"
                    "Cin: CO2 emisyonlari nufustan cok daha hizli artti.\n"
                    "Hindistan: Nufus ve emisyonlar paralel artiyor.\n"
                    "ABD: Nufus artarken emisyonlar azaldi, bu da basarili bir ayrisma oldugunu gosteriyor."))
pdf.add_image("img/pop_vs_co2_China.png", tr("Cin: Nufus vs CO2"))
pdf.add_image("img/pop_vs_co2_India.png", tr("Hindistan: Nufus vs CO2"))
pdf.add_image("img/pop_vs_co2_United States.png", tr("ABD: Nufus vs CO2"))

pdf.chapter_title(tr("8. Nüfus Ölçeği ve Kişi Başına Emisyon Dinamikleri"))
pdf.chapter_body(tr("Grafik üzerindeki dağılım, nüfus büyüklüğü ile kişi başı emisyonlar arasında doğrudan bir ilişki olmadığını, ancak kalkınma modellerinin belirleyici olduğunu göstermektedir:\n"
                    "- Çin: Çok yüksek nüfusa sahip olmasına rağmen, kişi başı emisyonları orta seviyededir (Sanayileşme etkisi).\n"
                    "- Hindistan: En yüksek nüfusa sahip olmasına rağmen kişi başı emisyonları düşüktür.\n"
                    "- ABD: Nüfusu Çin'e göre düşük olmasına rağmen, kişi başı emisyonları çok yüksektir (Yüksek tüketim ve enerji yoğunluğu).\n"
                    "- Türkiye: Düşük nüfus ve orta seviye kişi başı emisyon ile gelişmekte olan ülke profilini yansıtmaktadır."))
pdf.add_image("img/population_vs_per_capita.png", tr("Nufus vs Kisi Basi"))

pdf.chapter_title(tr("9. Demografik Projeksiyonlar (2025-2028)"))
pdf.chapter_body(tr("Gelecek yillar icin nufus buyumesini tahmin ettim.\n"
                    "- Cin: Zirve yapip dususe gecmesi bekleniyor.\n"
                    "- Hindistan: Nufus artisi devam edecek.\n"
                    "- ABD ve Turkiye: Buyumeye devam etmesi bekleniyor.\n"
                    "- Rusya ve Almanya: Nispeten istikrarli kalmasi veya azalmasi bekleniyor."))
pdf.add_image("img/population_forecast.png", tr("Nufus Tahmini"))

pdf.chapter_title(tr("10. Nüfus Kaynaklı Emisyon Etki Analizi"))
pdf.chapter_body(tr("Sadece nufus buyumesinin CO2 uzerindeki etkisini modelledim.\n"
                    "- Bu projeksiyon, nufus ve CO2 arasindaki tarihsel iliskinin sabit kaldigini varsayar.\n"
                    "- Sapma: Bunu gercek CO2 tahminiyle karsilastirmak, ulkelerin emisyonlari nufus artisindan nerede basariyla ayirdigini gosterir."))
pdf.add_image("img/co2_impact_analysis.png", tr("CO2 Etki Analizi"))

pdf.chapter_title(tr("11. Fosil Yakıt Bağımlılığı ve Enerji Kaynakları Analizi"))
pdf.chapter_body(tr("Önerileri doğrulamak için CO2 emisyonlarının kaynağını (Kömür, Petrol, Gaz) analiz ettik. Veriler farklı enerji profillerini doğrulamaktadır:\n\n"
                    "- Çin: Kömüre aşırı bağımlı (>%70), endüstriyel kömür kullanımını hedefleyen yeşil büyüme stratejilerine ihtiyaç var.\n"
                    "- Hindistan: Kömür baskın enerji kaynağı, yenilenebilir enerjiye geçiş kritik.\n"
                    "- Rusya: Gaz ve Petrol ağırlıklı (>%80), fosil yakıt bağımlılığından uzaklaşma önerisini destekliyor.\n"
                    "- ABD: Petrol ve Gaz ağırlıklı karma bir profil, ulaşım ve ısınma kaynaklı emisyonları yansıtıyor.\n"
                    "- Türkiye: Kömür ve Gazın önemli bir payı var, yenilenebilir enerjiye geçiş ve verimlilik stratejisi şart.\n"
                    "- Almanya: Yenilenebilir enerjiye rağmen Kömür hala önemli bir faktör, kömürden çıkış stratejisini haklı çıkarıyor."))
pdf.add_image("img/fossil_fuel_mix.png", tr("Fosil Yakıt Emisyon Dağılımı"))

pdf.chapter_title(tr("12. Üretim ve Tüketim Temelli Emisyon Analizi"))
pdf.chapter_body(tr("Bu analiz, ülkelerin emisyonlarını kendi sınırları içinde mi ürettiğini yoksa ithalat yoluyla dışarıdan mı aldığını (karbon sızıntısı) gösterir.\n\n"
                    "- Çin ve Hindistan: Üretim emisyonları tüketimden yüksektir. Bu, dünyanın fabrikası olduklarını ve gelişmiş ülkeler için emisyon ihraç ettiklerini gösterir.\n"
                    "- ABD ve Almanya: Tüketim emisyonları üretimden yüksektir. Bu, emisyon yoğun ürünleri ithal ettikleri anlamına gelir (Karbon Sızıntısı).\n"
                    "- Türkiye: Üretim ve tüketim dengeli seyretmektedir."))
pdf.add_image("img/prod_vs_cons_China.png", tr("Çin: Üretim vs Tüketim"))
pdf.add_image("img/prod_vs_cons_United States.png", tr("ABD: Üretim vs Tüketim"))

pdf.chapter_title(tr("13. Karbon Yoğunluğu Analizi (CO2 / GSYİH)"))
pdf.chapter_body(tr("Ekonomik büyümenin ne kadar 'yeşil' olduğunu ölçer. Düşük karbon yoğunluğu, birim GSYİH başına daha az emisyon üretildiği anlamına gelir.\n\n"
                    "- Küresel Trend: Genel olarak karbon yoğunluğu düşmektedir, bu da teknolojinin geliştiğini ve enerji verimliliğinin arttığını gösterir.\n"
                    "- Çin: Hızlı bir düşüş trendindedir, ekonomisini modernize etmektedir.\n"
                    "- ABD ve Almanya: Düşük ve istikrarlı bir yoğunluğa sahiptir, gelişmiş ve verimli ekonomilerdir.\n"
                    "- Hindistan: Yoğunluk hala yüksektir ancak düşüş eğilimindedir."))
pdf.add_image("img/carbon_intensity_trend.png", tr("Karbon Yoğunluğu Trendi"))

pdf.chapter_title(tr("14. Stratejik Öneriler ve Emisyon Azaltım Senaryoları"))
pdf.chapter_body(tr("2028 projeksiyonları ve nüfus dinamikleri ışığında ülkelere özel stratejik öneriler:\n\n"
                    "Çin: Nüfusun zirve yapıp azalması ve emisyonların düşüş trendine girmesi bekleniyor. Öneri: Yenilenebilir enerji yatırımlarını artırarak bu düşüşü hızlandırın ve sanayide elektrifikasyona geçin.\n\n"
                    "Hindistan: Nüfus artışı devam ediyor. Model düşüş öngörse de artan enerji talebi risk oluşturuyor. Öneri: Kömürden uzaklaşarak güneş ve rüzgar enerjisi kapasitesini artırın.\n\n"
                    "ABD: Nüfus artışı ve modeldeki emisyon artış öngörüsü dikkat çekici. Öneri: Kişi başı emisyonları düşürmek için enerji verimliliğini artırın ve fosil yakıt sübvansiyonlarını kaldırın.\n\n"
                    "Türkiye: Nüfus artışı sürüyor. Model düşüş öngörüyor. Öneri: Enerji ithalatını azaltmak için yerli yenilenebilir kaynaklara (Güneş, Rüzgar) yönelin.\n\n"
                    "Almanya: Nüfus durağan ancak modelde emisyon artışı riski görülüyor. Öneri: Kömürden çıkış planını hızlandırın ve sanayide hidrojen kullanımını teşvik edin.\n\n"
                    "Rusya: Nüfus durağan. Emisyon artış riski var. Öneri: Ekonomiyi fosil yakıt ihracatından çeşitlendirin ve enerji verimliliğine odaklanın."))

pdf.chapter_title(tr("15. Sonuç ve Özet"))
pdf.chapter_body(tr("Küresel CO2 Geleceği: Mevcut trendler, acil müdahale edilmediği takdirde emisyonların artmaya devam edeceğini göstermektedir.\n\n"
                    "Ülke Bazlı Çıkarımlar:\n"
                    "- Çin, Hindistan ve Türkiye: Büyüme odaklı emisyon artışı devam etmektedir.\n"
                    "- ABD ve Almanya: Verimlilik ve politika değişiklikleri ile emisyonları düşürmeyi başarmışlardır.\n"
                    "- Rusya: Fosil yakıt bağımlılığı nedeniyle durağan bir seyir izlemektedir.\n\n"
                    "En Kritik Riskler: İklim değişikliğine bağlı aşırı hava olayları, kaynak kıtlığı ve halk sağlığı üzerindeki baskılar artmaktadır.\n\n"
                    "Etkili Politika Önerileri: Yenilenebilir enerjiye geçişin hızlandırılması, döngüsel ekonomi modellerinin benimsenmesi ve uluslararası işbirliğinin güçlendirilmesi gerekmektedir."))

pdf.output("CO2_Analiz_Raporu.pdf")
print("PDF olusturuldu: CO2_Analiz_Raporu.pdf")

from fpdf import FPDF
import os

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

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, tr('Nature Pollution - Analiz Raporu Sunum Rehberi'), 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, tr(f'Sayfa {self.page_no()}'), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, tr(title), 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, tr(body))
        self.ln()

def create_presentation_guide():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    pdf.chapter_body(
        "Bu rehber, 'CO2_Analiz_Raporu.pdf' dosyasını sunarken kullanacağınız hazır konuşma metinlerini ve "
        "teknik açıklamaları içerir. 'Sunum Notu' kısımlarını doğrudan okuyabilir veya ezberleyebilirsiniz."
    )

    # 0. Veri Hikayesi
    pdf.chapter_title("Giriş: Veri Hikayesi ve Değişkenler")
    pdf.chapter_body(
        "Sunum Notu: Analizimize başlamadan önce, kullandığımız verinin hikayesinden bahsetmek istiyorum. "
        "Verilerimiz, 'Our World in Data' platformundan alınmıştır ve Sanayi Devrimi'nden günümüze kadar olan süreci kapsar. "
        "Biz bu çalışmada, sadece CO2'yi değil, onu etkileyen Nüfus, GSYİH ve Enerji Tüketimi gibi değişkenleri de inceledik. "
        "Amacımız sadece 'ne kadar kirlendiğimizi' değil, 'neden kirlendiğimizi' anlamaktır.\n"
        "Teknoloji: Domain Knowledge (Alan Bilgisi).\n"
        "Neden?: İzleyiciye bağlamı (context) vermek için."
    )

    # 1. Veri ve Metodoloji
    pdf.chapter_title("Rapor Girişi: Veri ve Metodoloji")
    pdf.chapter_body(
        "Sunum Notu: Verilerimiz, uluslararası alanda güvenilirliği kabul görmüş 'Our World in Data' platformundan alınmıştır. "
        "Bu veri seti, bilimsel çalışmalarda referans olarak kullanılan standart bir kaynaktır.\n"
        "Teknoloji: Pandas kütüphanesi.\n"
        "Neden?: Ham veri setinde eksik yıllar vardı. Pandas'ın 'interpolate()' fonksiyonunu kullanarak bu boşlukları doğrusal artış mantığıyla doldurduk. Böylece grafiklerde kopukluk olmadı."
    )

    # 2. Model Performansı
    pdf.chapter_title("Rapor Bölümü: Model Performansı")
    pdf.chapter_body(
        "Sunum Notu: Modelimizin güvenilirliğini test etmek için veriyi Eğitim ve Test olarak ikiye ayırdık. "
        "Elde ettiğimiz yüksek R2 skoru, modelin %90'ın üzerinde bir başarıyla gerçek hayat verilerini temsil ettiğini göstermektedir.\n"
        "Teknoloji: Scikit-learn (R2 Score, RMSE).\n"
        "Neden?: Tahmin yapmadan önce modeli test etmeliyiz. Veriyi 2000-2018 (Eğitim) ve 2019-2024 (Test) olarak ikiye böldük."
    )

    # 3. Küresel CO2 Tarihsel Gelişimi
    pdf.chapter_title("1. Küresel CO2 Emisyonlarının Tarihsel Gelişimi")
    pdf.chapter_body(
        "Sunum Notu: Bu grafik, küresel ısınmanın duraksamadan devam ettiğini net bir şekilde ortaya koyuyor. "
        "2000 yılından günümüze emisyonlardaki istikrarlı artış, sorunun küresel boyutunu gözler önüne seriyor.\n"
        "Teknoloji: Matplotlib ve Seaborn (Lineplot).\n"
        "Neden?: Karmaşık sayıları tek bir çizgiyle özetlemek için. Grafikteki yukarı yönlü eğim, küresel ısınmanın durmadığını kanıtlıyor."
    )

    # 4. Ülke Bazlı Profiller
    pdf.chapter_title("2. Ülke Bazlı Emisyon Profilleri")
    pdf.chapter_body(
        "Sunum Notu: Dünya genelini analiz etmek yerine stratejik öneme sahip 6 ülkeye odaklandık. "
        "Çin'in son 20 yıldaki dik artışı ile ABD ve Avrupa'nın düşüş trendi, küresel dengelerin nasıl değiştiğini gösteriyor.\n"
        "Teknoloji: Pandas Filtering & Grouping.\n"
        "Neden?: Tüm dünyayı analiz etmek yerine stratejik 6 ülkeyi (Çin, ABD, Hindistan vb.) filtreledik. Bu sayede kıyaslama yapmak kolaylaştı."
    )

    # 5. Korelasyon Analizi
    pdf.chapter_title("3. Emisyon Sürücüleri: Korelasyon Analizi")
    pdf.chapter_body(
        "Sunum Notu: Isı haritamız, CO2 artışının en çok Nüfus ve GSYİH ile ilişkili olduğunu kanıtlıyor. "
        "Kırmızı alanlar bu güçlü ilişkiyi temsil eder; yani nüfus ve ekonomi büyüdükçe emisyon kaçınılmaz olarak artıyor.\n"
        "Teknoloji: Seaborn Heatmap (Isı Haritası).\n"
        "Neden?: Değişkenler arasındaki ilişkiyi renklerle göstermek için. Bu analiz, tahmin modelimizde neden 'Nüfus' verisini kullandığımızı haklı çıkarır."
    )

    # 6. Gelecek Projeksiyonları
    pdf.chapter_title("4. & 5. Gelecek Projeksiyonları (2025-2028)")
    pdf.chapter_body(
        "Sunum Notu: 2028 yılına kadar olan tahminlerimiz, kesik çizgilerle gösterilmiştir. "
        "Modelimiz, mevcut politikalar değişmezse Çin ve Hindistan kaynaklı artışın devam edeceğini, ancak Batı'da düşüşün süreceğini öngörüyor.\n"
        "Teknoloji: Scikit-learn Polynomial Regression (Polinom Regresyon).\n"
        "Neden?: CO2 emisyonları düz bir çizgi (Lineer) şeklinde artmaz, dalgalıdır. Polinom regresyon bu eğrisel hareketi yakalayabilir."
    )

    # 7. Kişi Başına Emisyon
    pdf.chapter_title("6. Nüfus Yoğunluğu ve Kişi Başına Düşen Emisyonlar")
    pdf.chapter_body(
        "Sunum Notu: Toplamda en çok kirleten Çin olsa da, kişi başına düşen emisyonda ABD hala liderdir. "
        "Bu durum, gelişmiş ülkelerin bireysel tüketim alışkanlıklarının çevreye daha fazla zarar verdiğini gösterir.\n"
        "Teknoloji: Feature Engineering (Özellik Mühendisliği).\n"
        "Neden?: Toplam emisyon yanıltıcı olabilir. Toplam emisyonu nüfusa bölerek (CO2 / Population) adil bir kıyaslama metriği ürettik."
    )

    # 8. Demografik Büyüme vs Emisyon
    pdf.chapter_title("7. Demografik Büyüme ve Emisyon İlişkisi")
    pdf.chapter_body(
        "Sunum Notu: Bu grafik, 'Decoupling' yani ayrışma başarısını gösterir. "
        "ABD ve Almanya'da nüfus artmasına rağmen emisyonların azalması, doğru politikalarla büyümenin çevreyi kirletmeden de mümkün olduğunun kanıtıdır.\n"
        "Teknoloji: Data Normalization (Endeksleme).\n"
        "Neden?: Nüfus (milyar) ve CO2 (milyon ton) farklı birimlerdir. İkisini de başlangıç yılında 100'e eşitleyerek artış hızlarını kıyasladık."
    )

    # 9. Nüfus Ölçeği vs Kişi Başı
    pdf.chapter_title("8. Nüfus Ölçeği ve Kişi Başına Emisyon Dinamikleri")
    pdf.chapter_body(
        "Sunum Notu: Nüfus büyüklüğü ile kirlilik arasında doğrudan bir bağ yoktur. "
        "Çin, hem çok kalabalık hem de sanayileştiği için kişi başı emisyonu artmaktadır, bu da onu benzersiz bir örnek yapmaktadır.\n"
        "Teknoloji: Scatter Plot (Dağılım Grafiği).\n"
        "Neden?: Ülkelerin gelişmişlik seviyelerini gruplamak için."
    )

    # 10. Demografik Projeksiyonlar
    pdf.chapter_title("9. Demografik Projeksiyonlar (2025-2028)")
    pdf.chapter_body(
        "Sunum Notu: CO2 tahminimizin temeli nüfustur. Çin nüfusunun zirve yapıp azalmaya başlayacak olması, "
        "gelecekte emisyonların da doğal olarak düşüşe geçebileceğinin en güçlü sinyalidir.\n"
        "Teknoloji: Time Series Forecasting (Zaman Serisi Tahmini).\n"
        "Neden?: CO2 tahmin modelimizin ana girdisi nüfustur. Önce nüfusu tahmin ettik ki, bu veriyi CO2 modeline besleyebilelim."
    )

    # 11. Nüfus Kaynaklı Etki
    pdf.chapter_title("10. Nüfus Kaynaklı Emisyon Etki Analizi")
    pdf.chapter_body(
        "Sunum Notu: Eğer teknoloji hiç gelişmeseydi emisyonlar çok daha yüksek olacaktı. "
        "Gerçek verinin simülasyondan düşük çıkması, enerji verimliliği ve yeşil teknolojilerin işe yaradığını ispatlıyor.\n"
        "Teknoloji: Simulation (Simülasyon).\n"
        "Neden?: Teknolojinin ve yeşil enerjinin etkisini ölçmek için."
    )

    # 12. Fosil Yakıt Analizi
    pdf.chapter_title("11. Fosil Yakıt Bağımlılığı")
    pdf.chapter_body(
        "Sunum Notu: Sorunun kaynağı ülkeye göre değişiyor: Çin'de kömür, Rusya'da doğalgaz, ABD'de ise petrol baskın. "
        "Bu veri, her ülkeye neden farklı bir çözüm önerdiğimizin dayanağıdır.\n"
        "Teknoloji: Data Aggregation (Veri Toplulaştırma).\n"
        "Neden?: Sorunun kaynağını bulmadan çözüm öneremeyiz."
    )

    # 13. Üretim vs Tüketim
    pdf.chapter_title("12. Üretim ve Tüketim Temelli Emisyon Analizi")
    pdf.chapter_body(
        "Sunum Notu: Bu grafik, 'Karbon Sızıntısı' kavramını açıklar. Gelişmiş ülkeler (ABD, Almanya) emisyonlarını düşürmüş gibi görünse de, aslında kirli üretimi Çin gibi ülkelere taşımışlardır. "
        "Yani kendi topraklarında temizler ama tükettikleri ürünler başka yerde dünyayı kirletmeye devam ediyor.\n"
        "Teknoloji: Comparative Analysis (Karşılaştırmalı Analiz).\n"
        "Neden?: Emisyon sorumluluğunun sadece üreticiye değil, tüketiciye de ait olduğunu göstermek için."
    )

    # 14. Karbon Yoğunluğu
    pdf.chapter_title("13. Karbon Yoğunluğu Analizi (CO2 / GSYİH)")
    pdf.chapter_body(
        "Sunum Notu: Bu grafik ekonominin ne kadar 'Yeşil' olduğunu gösterir. Çin'in grafiğindeki sert düşüş, "
        "ekonomisi büyürken artık daha az enerji harcadığını ve teknolojisini modernize ettiğini kanıtlıyor.\n"
        "Teknoloji: Ratio Analysis (Oran Analizi).\n"
        "Neden?: Sadece toplam emisyona bakmak haksızlık olur; ülkenin parayı ne kadar temiz kazandığına da bakmalıyız."
    )

    # 15. Stratejik Öneriler
    pdf.chapter_title("14. Stratejik Öneriler")
    pdf.chapter_body(
        "Sunum Notu: Veri analizimiz sonucunda; Çin'e kömürü bırakmasını, ABD'ye ise bireysel tüketimi azaltmasını öneriyoruz. "
        "Bu öneriler kişisel görüş değil, doğrudan emisyon kaynakları verisine dayalıdır.\n"
        "Teknoloji: Data-Driven Insight (Veri Odaklı İçgörü).\n"
        "Neden?: Analizin bir sonuca varması gerekir."
    )

    # 16. Sonuç
    pdf.chapter_title("15. Sonuç ve Özet")
    pdf.chapter_body(
        "Sunum Notu: Sonuç olarak, 2028 projeksiyonları küresel bir dönüm noktasında olduğumuzu gösteriyor. "
        "Gelişmiş ülkeler emisyonu düşürmeyi başardı, şimdi sıra gelişmekte olan ülkelerin temiz enerjiye geçişini hızlandırmakta."
    )

    pdf.output("Sunum_Rehberi.pdf")
    print("Sunum rehberi oluşturuldu: Sunum_Rehberi.pdf")

if __name__ == "__main__":
    create_presentation_guide()

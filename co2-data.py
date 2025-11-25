import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
from datetime import date
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)


def load():
    """
    Veri setini yükler. Dosya bulunamazsa alternatif yolu dener.
    
    Returns:
        pd.DataFrame: Yüklenen veri seti.
    """
    try:
        data = pd.read_csv("Datasets/owid-co2-data.csv")
    except FileNotFoundError:
        # Eğer ana dizinden çalıştırılıyorsa alternatif yolu dener
        data = pd.read_csv("Nature-Pollution-main/Datasets/owid-co2-data.csv")
    return data

df = load()

def clean_and_balance_data(data):
    """
    Veri setindeki eksik değerleri temizler ve enterpolasyon ile doldurur.
    
    Args:
        data (pd.DataFrame): Ham veri seti.
        
    Returns:
        pd.DataFrame: Temizlenmiş ve doldurulmuş veri seti.
    """
    print("\n--- Data Quality Report (Before Cleaning) ---")
    print("Missing Values (%):")
    print(data[['co2', 'population', 'gdp']].isnull().mean() * 100)
    
    # Önemli ülkeler için veri sayısını kontrol et
    countries_check = ['China', 'United States', 'Russia', 'Turkey', 'Germany']
    print("\nData Points per Country (Selected):")
    print(data[data['country'].isin(countries_check)]['country'].value_counts())

    # Her ülke için eksik değerleri enterpolasyon ile doldur
    # Doğru enterpolasyon için ülke ve yıla göre sırala
    data = data.sort_values(['country', 'year'])
    
    # Ülkeye göre grupla ve sayısal sütunları enterpolasyon yap
    # Enterpolasyon sonrası boş kalan kenar değerler için ileri ve geri doldurma yöntemlerini kullan
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Gruplama işlemini güvenli bir şekilde uygulamak için apply kullan
    def fill_group(group):
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear', limit_direction='both')
        return group

    data = data.groupby('country', group_keys=False).apply(fill_group)
    
    print("\n--- Data Quality Report (After Interpolation) ---")
    print("Missing Values (%):")
    print(data[['co2', 'population', 'gdp']].isnull().mean() * 100)
    
    return data

df = clean_and_balance_data(df)

# img klasörünün var olduğundan emin ol
output_dir = "img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Standart Renkleri Tanımla
COUNTRY_COLORS = {
    'China': '#E74C3C',        # Red
    'United States': '#3498DB', # Blue
    'Germany': '#F1C40F',       # Yellow
    'Russia': '#8E44AD',        # Purple
    'Turkey': '#E67E22'         # Orange
}

# 1. Yıllara Göre Genel CO2 Artışı
print("--- General CO2 Increase Over Years ---")
yearly_co2 = df.groupby('year')['co2'].mean()
print(yearly_co2.tail())

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='year', y='co2', estimator='mean', errorbar=None, color='black') # Küresel trend siyah renkte
plt.title('Average Global CO2 Emissions Over Years')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/global_co2_trend.png')
print(f"Saved {output_dir}/global_co2_trend.png")

# 2. Ülkeye Özgü Analiz
print("\n--- Country-Specific Analysis ---")
countries = ['China', 'United States', 'Russia', 'Turkey', 'Germany']
df_countries = df[df['country'].isin(countries)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_countries, x='year', y='co2', hue='country', palette=COUNTRY_COLORS)
plt.title('CO2 Emissions by Country')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend(title='Country')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/country_co2_trend.png')
print(f"Saved {output_dir}/country_co2_trend.png")

# 3. Korelasyon Analizi
print("\n--- Correlation Analysis ---")
# CO2 ve kalkınma ile ilgili sayısal sütunları seç
cols_to_corr = ['co2', 'gdp', 'population', 'energy_per_capita', 'co2_per_capita']
# Daha alakalı veriler elde etmek için yakın yılları filtrele ve korelasyon için boş değerleri at
df_corr = df[df['year'] > 1990][cols_to_corr].dropna()

corr_matrix = df_corr.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Post-1990)')
plt.savefig(f'{output_dir}/correlation_matrix.png')
print(f"Saved {output_dir}/correlation_matrix.png")

# 4. Gelişmiş Analiz ve Tahmin
print("\n--- Advanced Analysis & Prediction ---")

def predict_co2(data, country_name=None):
    """
    Gelecekteki CO2 emisyonlarını tahmin etmek için Polinom Regresyon modeli kullanır.
    
    Args:
        data (pd.DataFrame): Veri seti.
        country_name (str, optional): Tahmin yapılacak ülke ismi. None ise küresel tahmin yapılır.
        
    Returns:
        tuple: Eğitim verisi, gelecek yıllar, tahminler, model, alt güven aralığı, üst güven aralığı.
    """
    # Eğer belirtildiyse belirli bir ülke için, aksi takdirde küresel veriler için filtrele
    if country_name:
        df_subset = data[data['country'] == country_name]
        title_suffix = f" ({country_name})"
    else:
        df_subset = data.groupby('year')['co2'].mean().reset_index()
        title_suffix = " (Global Average)"
    
    # Yakın dönem trendlerini yakalamak için 2000-2024 arası verileri eğitim için kullan
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)].dropna(subset=['co2'])
    
    if len(df_train) < 5:
        print(f"Not enough data for {title_suffix}")
        return None, None, None, None, None, None # Fixed unpacking error

    # Doğrusal olmayan trendleri yakalamak için Polinom Regresyon (2. Derece) kullan
    X = df_train[['year']]
    y = df_train['co2']
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Önümüzdeki 6 yıl (2025-2030) için tahmin yap
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    predictions = model.predict(future_years_poly)
    
    # Güven Aralıklarını Hesapla
    # 1. Ortalama Kare Hatasını (MSE) Hesapla
    y_pred_train = model.predict(X_poly)
    mse = np.sum((y - y_pred_train) ** 2) / (len(y) - X_poly.shape[1])
    
    # 2. (X^T X)^-1 değerini hesapla
    xtx_inv = np.linalg.inv(np.dot(X_poly.T, X_poly))
    
    # 3. Her bir gelecek noktası için varyansı hesapla
    # Güven Aralığı (Ortalama) için Varyans formülü: Var = MSE * (x_new^T * (X^T X)^-1 * x_new)
    # Tahmin Aralığı (Veri) için Varyans formülü: Var = MSE * (1 + x_new^T * (X^T X)^-1 * x_new)
    # Trend çizgisi belirsizliği için Güven Aralığını kullanıyoruz
    
    ci_lower = []
    ci_upper = []
    
    import scipy.stats as stats
    t_score = stats.t.ppf(0.975, df=len(y) - X_poly.shape[1]) # 95% CI
    
    for i in range(len(future_years_poly)):
        x_new = future_years_poly[i]
        var = mse * np.dot(np.dot(x_new, xtx_inv), x_new.T)
        se = np.sqrt(var)
        margin = t_score * se
        ci_lower.append(predictions[i] - margin)
        ci_upper.append(predictions[i] + margin)
        
    return df_train, future_years, predictions, model, np.array(ci_lower), np.array(ci_upper)

def evaluate_model(data):
    """
    Modelin performansını RMSE, MAE ve R2 metrikleri ile değerlendirir.
    
    Args:
        data (pd.DataFrame): Veri seti.
    """
    print("\n--- Model Evaluation (Global CO2) ---")
    # Küresel Verileri Hazırla
    df_subset = data.groupby('year')['co2'].mean().reset_index()
    df_train_full = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)].dropna(subset=['co2'])
    
    X = df_train_full[['year']]
    y = df_train_full['co2']
    
    # Veriyi %80 eğitim ve %20 test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeli Eğit
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    
    # Tahmin Yap
    y_pred = model.predict(X_test)
    
    # Metrikleri Hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_split": 0.8,
        "test_split": 0.2
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

# Önce Modeli Değerlendir
evaluate_model(df)

# Küresel Tahmin
df_train_global, future_years, pred_global, model_global, ci_lower_global, ci_upper_global = predict_co2(df)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train_global, x='year', y='co2', label='Historical (2000-2024)', color='black')
plt.plot(future_years, pred_global, color='red', linestyle='--', label='Prediction (2025-2030)')
plt.fill_between(future_years.flatten(), ci_lower_global, ci_upper_global, color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Global CO2 Emissions Forecast')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/global_forecast.png')
print(f"Saved {output_dir}/global_forecast.png")

# Ülke Bazlı Tahminler
plt.figure(figsize=(14, 7))
for country in countries:
    df_train, future_years, preds, model, ci_lower, ci_upper = predict_co2(df, country)
    if preds is not None:
        # Geçmiş verilerin son kısmını ve tahminleri çizdir
        color = COUNTRY_COLORS.get(country, 'gray')
        plt.plot(df_train['year'], df_train['co2'], label=f'{country} Historical', color=color, alpha=0.6)
        plt.plot(future_years, preds, linestyle='--', label=f'{country} Prediction', color=color, linewidth=2)
        plt.fill_between(future_years.flatten(), ci_lower, ci_upper, color=color, alpha=0.1)
        
        # Trend Analizi
        # Polinom için, son tahmin ile son geçmiş veri arasındaki farka veya sondaki eğime bakabiliriz
        # Basit yaklaşım: 2030 tahmininin 2024 verisinden (veya son mevcut veriden) büyük olup olmadığını kontrol et
        last_hist = df_train['co2'].iloc[-1]
        last_pred = preds[-1]
        
        trend = "Increasing" if last_pred > last_hist else "Decreasing"
        print(f"{country}: Trend is {trend} (2030 Pred: {last_pred:.2f} vs Last Hist: {last_hist:.2f})")

plt.title('CO2 Emissions Forecast by Country (2025-2030)')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/country_forecasts.png')
print(f"Saved {output_dir}/country_forecasts.png")

# 5. Sürücü Analizi ve Öneriler
print("\n--- Driver Analysis & Recommendations ---")
# Her ülke için CO2 ile en çok neyin ilişkili olduğunu analiz eder ve özel tavsiyeler verir
for country in countries:
    country_data = df[df['country'] == country].dropna(subset=['co2', 'gdp', 'energy_per_capita', 'population'])
    if len(country_data) > 10:
        # Sürücüleri bulmak için basit korelasyon
        corr = country_data[['co2', 'gdp', 'energy_per_capita', 'population']].corr()['co2']
        
        print(f"\nReport for {country}:")
        print(f"  - CO2 Correlation with GDP: {corr['gdp']:.2f}")
        print(f"  - CO2 Correlation with Energy: {corr['energy_per_capita']:.2f}")
        
        # Öneriler
        if corr['gdp'] > 0.9:
            print("  -> Recommendation: Focus on decoupling economic growth from emissions (Green Growth).")
        if corr['energy_per_capita'] > 0.9:
            print("  -> Recommendation: High energy dependency. Prioritize renewable energy transition and efficiency.")
        if corr['population'] > 0.9:
            print("  -> Recommendation: Population growth is a major driver. Focus on sustainable urban planning.")
            
# 6. Azaltım Senaryoları
print("\n--- Reduction Scenarios ---")
# 2050 yılına kadar 2024 seviyelerinin %50'sine ulaşmak için gereken azaltımı hesapla
target_year = 2050
current_year = 2024
years_remaining = target_year - current_year

for country in countries:
    country_data = df[(df['country'] == country) & (df['year'] == current_year)]
    if not country_data.empty:
        current_co2 = country_data['co2'].values[0]
        target_co2 = current_co2 * 0.5
        
        # Bileşik Yıllık Azaltım Oranı
        required_reduction = (1 - (target_co2 / current_co2) ** (1 / years_remaining)) * 100
        print(f"{country}: To halve emissions by 2050, needs {required_reduction:.2f}% annual reduction.")

# 7. Kişi Başına CO2 Analizi
print("\n--- CO2 per Capita Analysis ---")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_countries, x='year', y='co2_per_capita', hue='country', palette=COUNTRY_COLORS)
plt.title('CO2 Emissions per Capita by Country')
plt.ylabel('CO2 per Capita (Tonnes)')
plt.xlabel('Year')
plt.legend(title='Country')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/co2_per_capita_trend.png')
print(f"Saved {output_dir}/co2_per_capita_trend.png")

# 8. Nüfus ve CO2 Büyüme Analizi
print("\n--- Population vs CO2 Growth Analysis ---")
# Son 20 yılın (yaklaşık 2004-2024) büyüme oranlarını hesapla
start_year_growth = 2004
end_year_growth = 2024

for country in countries:
    country_data = df[(df['country'] == country) & (df['year'] >= start_year_growth) & (df['year'] <= end_year_growth)].sort_values('year')
    
    if not country_data.empty:
        # Karşılaştırma için başlangıç yılını 100 olarak normalize et
        base_pop = country_data['population'].iloc[0]
        base_co2 = country_data['co2'].iloc[0]
        
        country_data['pop_index'] = (country_data['population'] / base_pop) * 100
        country_data['co2_index'] = (country_data['co2'] / base_co2) * 100
        
        # Çift Eksenli Grafik
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # CO2 için ülke rengini, kontrast oluşturmak için Nüfus için nötr bir renk (örn. gri/siyah) kullan
        # Kullanıcı: "Ülke renkleri her grafik için sabit olmalı" dedi.
        # Ancak burada 1 ülke için 2 çizgimiz var.
        # CO2 (ana konu) için Ülke Rengini ve Nüfus için standart bir renk (örn. Siyah kesikli) kullanalım
        
        color_co2 = COUNTRY_COLORS.get(country, 'tab:red')
        color_pop = 'black'
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('CO2 Emissions (Index 2004=100)', color=color_co2)
        ax1.plot(country_data['year'], country_data['co2_index'], color=color_co2, label='CO2 Growth', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color_co2)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.set_ylabel('Population (Index 2004=100)', color=color_pop)  # we already handled the x-label with ax1
        ax2.plot(country_data['year'], country_data['pop_index'], color=color_pop, linestyle='--', label='Population Growth')
        ax2.tick_params(axis='y', labelcolor=color_pop)
        
        plt.title(f'{country}: Population vs CO2 Growth (Indexed)')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'{output_dir}/pop_vs_co2_{country}.png')
        print(f"Saved {output_dir}/pop_vs_co2_{country}.png")
        plt.close()

# 9. Nüfusa Göre Kişi Başına Değişim
print("\n--- Per Capita Change relative to Population ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_countries[df_countries['year'] > 1990], x='population', y='co2_per_capita', hue='country', palette=COUNTRY_COLORS)
plt.title('Population vs CO2 per Capita (Post-1990)')
plt.ylabel('CO2 per Capita (Tonnes)')
plt.xlabel('Population')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/population_vs_per_capita.png')
print(f"Saved {output_dir}/population_vs_per_capita.png")

# 10. Nüfus Tahmini ve CO2 Etki Analizi
print("\n--- Population Prediction & CO2 Impact Analysis ---")

def predict_population(data, country_name):
    """
    Gelecekteki nüfusu tahmin etmek için Doğrusal Regresyon modeli kullanır.
    
    Args:
        data (pd.DataFrame): Veri seti.
        country_name (str): Tahmin yapılacak ülke ismi.
        
    Returns:
        tuple: Eğitim verisi, gelecek yıllar, tahminler.
    """
    df_subset = data[data['country'] == country_name].dropna(subset=['population'])
    # Eğitim için 2000-2024 arası verileri kullan
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)]
    
    if len(df_train) < 5:
        return None, None, None

    X = df_train[['year']]
    y = df_train['population']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Önümüzdeki 10 yıl (2025-2035) için tahmin yap
    future_years = np.arange(2025, 2036).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return df_train, future_years, predictions

def predict_co2_from_population(data, country_name, future_pop):
    """
    Tahmin edilen nüfus verilerine dayanarak gelecekteki CO2 emisyonlarını tahmin eder.
    
    Args:
        data (pd.DataFrame): Veri seti.
        country_name (str): Ülke ismi.
        future_pop (np.array): Gelecekteki nüfus tahminleri.
        
    Returns:
        np.array: Tahmin edilen CO2 emisyonları.
    """
    df_subset = data[data['country'] == country_name].dropna(subset=['population', 'co2'])
    # 2000-2024 arası verileri kullan
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)]
    
    if len(df_train) < 5:
        return None
        
    X = df_train[['population']]
    y = df_train['co2']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Gelecekteki nüfusa dayalı olarak CO2 tahmini yap
    # future_pop, önceki adımdan gelen bir numpy dizisidir
    future_co2 = model.predict(future_pop.reshape(-1, 1))
    
    return future_co2

# Görselleştirme: Nüfus Tahmini
plt.figure(figsize=(14, 7))
for country in countries:
    df_train, future_years, pop_preds = predict_population(df, country)
    if pop_preds is not None:
        color = COUNTRY_COLORS.get(country, 'gray')
        plt.plot(df_train['year'], df_train['population'], label=f'{country} Historical', color=color, alpha=0.6)
        plt.plot(future_years, pop_preds, linestyle='--', label=f'{country} Forecast', color=color, linewidth=2)

plt.title('Population Forecast (2025-2035)')
plt.ylabel('Population')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/population_forecast.png')
print(f"Saved {output_dir}/population_forecast.png")

# Görselleştirme: CO2 Etki Analizi (Nüfus Kaynaklı vs Zaman Serisi Tahmini)
# Sadece Nüfus büyümesine dayalı CO2 tahminini, Zamana dayalı tahminle karşılaştıracağız
plt.figure(figsize=(14, 7))

for country in countries:
    # 1. Nüfus Tahminini Al
    _, future_years, pop_preds = predict_population(df, country)
    
    if pop_preds is not None:
        # 2. Nüfus Tahminine dayalı olarak CO2 tahmini yap
        co2_from_pop = predict_co2_from_population(df, country, pop_preds)
        
        if co2_from_pop is not None:
            # Netlik için sadece tahmin kısmını çizdir veya üst üste bindir
            # "Etki" çizgisini çizelim (Eğer nüfus trendini takip ederse CO2)
            color = COUNTRY_COLORS.get(country, 'gray')
            plt.plot(future_years, co2_from_pop, linestyle=':', marker='.', label=f'{country} (Pop-Driven CO2)', color=color)

plt.title('Projected CO2 Emissions based ONLY on Population Growth (2025-2035)')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/co2_impact_analysis.png')
print(f"Saved {output_dir}/co2_impact_analysis.png")

# 11. Fosil Yakıt Kaynakları Analizi
print("\n--- Fossil Fuel Sources Analysis ---")
# En son mevcut yıl için Kömür, Petrol ve Gazın CO2 emisyonlarındaki payını analiz et
fuel_cols = ['coal_co2', 'oil_co2', 'gas_co2']
# Sütunların var olup olmadığını kontrol et
existing_fuel_cols = [c for c in fuel_cols if c in df.columns]

if existing_fuel_cols:
    plt.figure(figsize=(12, 8))
    
    # Her ülke için son yıl verilerini al
    last_year_data = []
    years_used = []
    
    for country in countries:
        country_df = df[df['country'] == country].sort_values('year')
        if not country_df.empty:
            # Bu sütunlar için geçerli veriye sahip son satırı al
            valid_row = country_df.dropna(subset=existing_fuel_cols).tail(1)
            if not valid_row.empty:
                last_year_data.append(valid_row)
                years_used.append(valid_row['year'].iloc[0])
    
    if last_year_data:
        df_fuel = pd.concat(last_year_data)
        
        # Yıl etiketini belirle
        unique_years = sorted(list(set(years_used)))
        if len(unique_years) == 1:
            year_label = str(unique_years[0])
        else:
            year_label = f"{min(unique_years)}-{max(unique_years)}"
        
        # Yüzdelik payı hesapla
        # Not: Bu sütunlar genellikle mutlak değerlerdir (ton). Toplam fosil CO2'yi elde etmek için bunları toplamamız gerekir
        # "Karışımı" göstermek için bu üç kaynağın toplamındaki payı hesaplayalım
        df_fuel['total_fossil'] = df_fuel[existing_fuel_cols].sum(axis=1)
        
        for col in existing_fuel_cols:
            df_fuel[f'{col}_share'] = (df_fuel[col] / df_fuel['total_fossil']) * 100
            
        # Yığılmış Çubuk Grafiği Çizimi
        plot_data = df_fuel.set_index('country')[[f'{c}_share' for c in existing_fuel_cols]]
        plot_data.columns = [c.replace('_co2', '').title() for c in existing_fuel_cols] # Gösterge için yeniden adlandır
        
        plot_data.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
        
        plt.title(f'Fossil Fuel CO2 Emission Mix (Year Used: {year_label})')
        plt.ylabel('Percentage Share (%)')
        plt.xlabel('Country')
        plt.legend(title='Fuel Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/fossil_fuel_mix.png')
        print(f"Saved {output_dir}/fossil_fuel_mix.png")
        
        # Rapor için belirli değerleri yazdır
        print("\nFossil Fuel Mix Data:")
        print(plot_data)
    else:
        print("No valid data found for fossil fuel analysis.")
else:
    print("Fossil fuel columns not found in dataset.")

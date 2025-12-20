import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

"""
Bu dosya, Nature-Pollution/co2-data.py'nin "time-safe" (lookahead leakage yok) versiyonudur.



Temel fark :
- Eksik değer doldurma / interpolasyon, train-test ayrımından SONRA yapılır.
  - Train: interpolate(linear, both) + ffill/bfill
  - Test: sadece ffill (gelecek bilgisi kullanılmaz), baştaki NaN -> train son değeri

Not:
- Bu dosya, zaman sızıntısını (lookahead) engeller.
- Ancak co2_per_capita ve co2_per_gdp gibi sütunlar "target leakage" olabilir (co2'yu içerir).
  Model FEATURES'ten bunları çıkarmak ayrı bir karardır.
"""


def load():
    """
    Veri setini yükler. Dosya bulunamazsa alternatif yolu dener.
    """
    try:
        data = pd.read_csv("Datasets/owid-co2-data.csv")
    except FileNotFoundError:
        data = pd.read_csv("Nature-Pollution/Datasets/owid-co2-data.csv")
    return data



FEATURES = [
    "year",
    "gdp",
    "population",
    "primary_energy_consumption",
    "energy_per_capita",
    "co2_per_capita",
    "co2_per_gdp",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2",
    "flaring_co2",
    "methane",
    "nitrous_oxide",
]


def clean_and_balance_data_for_eda(data: pd.DataFrame) -> pd.DataFrame:
    """
    (EDA/Görselleştirme amaçlı)  ülke bazında "both" interpolasyon uygular.
    Bu fonksiyon, model değerlendirme metrikleri için önerilmez; sadece grafik/EDA için tutulur.
    """
    print("\n--- Data Quality Report (Before Cleaning) ---")
    print("Missing Values (%):")
    print(data[["co2", "population", "gdp"]].isnull().mean() * 100)

    countries_check = ["China", "United States", "Russia", "Turkey", "Germany", "India"]
    print("\nData Points per Country (Selected):")
    print(data[data["country"].isin(countries_check)]["country"].value_counts())

    data = data.sort_values(["country", "year"])
    cols_to_interpolate = list(set(FEATURES + ["co2"]))
    cols_to_interpolate = [c for c in cols_to_interpolate if c in data.columns]

    def fill_group(group):
        group[cols_to_interpolate] = group[cols_to_interpolate].interpolate(
            method="linear", limit_direction="both"
        )
        return group

    data = data.groupby("country", group_keys=False).apply(fill_group)

    print("\n--- Data Quality Report (After Interpolation) ---")
    print("Missing Values (%):")
    print(data[["co2", "population", "gdp"]].isnull().mean() * 100)

    return data


def _build_global_avg(data: pd.DataFrame) -> pd.DataFrame:
    
    cols = list(set(FEATURES + ["co2"]))
    cols = [c for c in cols if c in data.columns and c != "year"]
    df_subset = data.groupby("year")[cols].mean(numeric_only=True).reset_index()
    df_subset = df_subset.sort_values("year")
    return df_subset


def _country_time_safe_impute_after_split(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    - Önce train/test split yapılmış olmalı.
    - Train: ülke içinde interpolate(both) + ffill/bfill
    - Test : ülke içinde sadece ffill; baştaki NaN -> o ülkenin train son değeri
    
    """
    tr = train_df.sort_values(["country", "year"]).copy()
    te = test_df.sort_values(["country", "year"]).copy()

    tr[cols] = tr[cols].replace([np.inf, -np.inf], np.nan)
    te[cols] = te[cols].replace([np.inf, -np.inf], np.nan)

    def fill_train(g: pd.DataFrame) -> pd.DataFrame:
        g[cols] = g[cols].interpolate(method="linear", limit_direction="both").ffill().bfill()
        return g

    tr = tr.groupby("country", group_keys=False).apply(fill_train)

    # test: sadece geçmiş (ffill)
    te = te.groupby("country", group_keys=False).apply(lambda g: g.assign(**{c: g[c].ffill() for c in cols}))

    # test başındaki NaN -> ülkenin train son değeri
    last_vals = tr.groupby("country")[cols].last()
    te = te.set_index("country")
    for c in cols:
        te[c] = te[c].fillna(last_vals[c])
    te = te.reset_index()

    return tr, te


def _time_safe_impute_after_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fill_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/Test split SONRASI imputasyon.
    - Train: interpolate(both) + ffill/bfill
    - Test : ffill (sadece geçmiş) + baştaki NaN -> train son değeri
    """
    tr = train_df.copy()
    te = test_df.copy()

    # inf guard (özellikle co2_per_gdp gibi oranlarda)
    tr[fill_cols] = tr[fill_cols].replace([np.inf, -np.inf], np.nan)
    te[fill_cols] = te[fill_cols].replace([np.inf, -np.inf], np.nan)

    tr[fill_cols] = tr[fill_cols].interpolate(method="linear", limit_direction="both").ffill().bfill()

    te[fill_cols] = te[fill_cols].ffill()
    for c in fill_cols:
        if te[c].isna().any():
            te[c] = te[c].fillna(tr[c].iloc[-1])

    return tr, te


def evaluate_model_multivariate_time_safe(data: pd.DataFrame) -> dict:
    """
    Orijinal evaluate_model_multivariate ile aynı model (LinearRegression) ve aynı split:
    - Train: 2000-2018
    - Test : 2019-2024

    Fark: Eksik değer doldurma split'ten sonra "time-safe" yapılır.
    """
    print("\n--- Model Evaluation (Multivariate Global, TIME-SAFE) ---")

    # 1) Split önce (ham country-level veri)
    train_raw = data[(data["year"] >= 2000) & (data["year"] <= 2018)].copy()
    test_raw = data[(data["year"] >= 2019) & (data["year"] <= 2024)].copy()

    # 2) Country-level time-safe imputasyon
    cols_for_country = [c for c in (FEATURES + ["co2"]) if c in data.columns and c not in ["year", "country"]]
    train_imp, test_imp = _country_time_safe_impute_after_split(train_raw, test_raw, cols=cols_for_country)

    # 3) Notebook'taki gibi global average üret
    df_train = _build_global_avg(train_imp)
    df_test = _build_global_avg(test_imp)

    model_cols = [c for c in FEATURES if c in df_train.columns]  # year dahil

    
    df_train = df_train.dropna(subset=["co2"] + model_cols)
    df_test = df_test.dropna(subset=["co2"] + model_cols)

    X_train = df_train[model_cols]
    y_train = df_train["co2"]
    X_test = df_test[model_cols]
    y_test = df_test["co2"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print("\nFeature Importance (Coefficients):")
    coef_df = pd.DataFrame({"Feature": model_cols, "Coefficient": model.coef_})
    coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
    print(coef_df.sort_values("Abs_Coef", ascending=False))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_period": "2000-2018",
        "test_period": "2019-2024",
        "model_type": "Multivariate Linear Regression (TIME-SAFE IMPUTE)",
    }

    with open("metrics_timesafe.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics_timesafe.json")

    return metrics


def forecast_features(data: pd.DataFrame, future_years: np.ndarray) -> pd.DataFrame:
    """
    her feature için year->feature polinom(2) ile tahmin.
    """
    forecasts = {}
    feature_cols = [c for c in FEATURES if c != "year" and c in data.columns]
    future_years_reshaped = future_years.reshape(-1, 1)

    for col in feature_cols:
        df_feat = data[["year", col]].dropna()
        if len(df_feat) < 5:
            last_val = df_feat[col].iloc[-1] if not df_feat.empty else 0
            forecasts[col] = np.full(len(future_years), last_val)
            continue

        X_feat = df_feat[["year"]]
        y_feat = df_feat[col]

        poly_feat = PolynomialFeatures(degree=2)
        X_poly_feat = poly_feat.fit_transform(X_feat)

        model_feat = LinearRegression()
        model_feat.fit(X_poly_feat, y_feat)

        future_poly = poly_feat.transform(future_years_reshaped)
        forecasts[col] = model_feat.predict(future_poly)

    return pd.DataFrame(forecasts, index=future_years.flatten())


def predict_co2_multivariate(data: pd.DataFrame, country_name: str | None = None):
   
    if country_name:
        df_subset = data[data["country"] == country_name].copy()
        title_suffix = f" ({country_name})"
    else:
        cols = list(set(FEATURES + ["co2"]))
        if "year" in cols:
            cols.remove("year")
        df_subset = data.groupby("year")[cols].mean(numeric_only=True).reset_index()
        title_suffix = " (Global Average)"

    model_cols = [c for c in FEATURES if c in df_subset.columns]
    df_train = df_subset[(df_subset["year"] >= 2000) & (df_subset["year"] <= 2024)].dropna(subset=["co2"] + model_cols)

    if len(df_train) < 10:
        print(f"Not enough data for {title_suffix}")
        return None, None, None, None, None, None

    X = df_train[model_cols]
    y = df_train["co2"]

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(2025, 2029)
    future_features_df = forecast_features(df_subset, future_years)
    future_features_df["year"] = future_years
    X_future = future_features_df[model_cols]
    predictions = model.predict(X_future)

    y_pred_train = model.predict(X)
    residuals = y - y_pred_train
    std_error = np.std(residuals)

    ci_lower, ci_upper = [], []
    for i in range(len(predictions)):
        margin = 1.96 * std_error * np.sqrt(i + 1)
        ci_lower.append(predictions[i] - margin)
        ci_upper.append(predictions[i] + margin)

    return df_train, future_years, predictions, model, np.array(ci_lower), np.array(ci_upper)


if __name__ == "__main__":
    df = load()

    
    evaluate_model_multivariate_time_safe(df)

   
    df_eda = clean_and_balance_data_for_eda(df.copy())

   
    output_dir = "img"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    COUNTRY_COLORS = {
        "China": "#E74C3C",  
        "United States": "#3498DB",  
        "Germany": "#F1C40F",  
        "Russia": "#8E44AD",  
        "Turkey": "#E67E22",  
        "India": "#2ECC71",  
    }

    # 1. Yıllara Göre Genel CO2 Artışı
    print("--- General CO2 Increase Over Years ---")
    yearly_co2 = df_eda.groupby("year")["co2"].mean()
    print(yearly_co2.tail())

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_eda, x="year", y="co2", estimator="mean", errorbar=None, color="black")
    plt.title("Average Global CO2 Emissions Over Years")
    plt.ylabel("CO2 Emissions (Million Tonnes)")
    plt.xlabel("Year")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/global_co2_trend.png")
    print(f"Saved {output_dir}/global_co2_trend.png")

    # 2. Ülkeye Özgü Analiz
    print("\n--- Country-Specific Analysis ---")
    countries = ["China", "United States", "Russia", "Turkey", "Germany", "India"]
    df_countries = df_eda[df_eda["country"].isin(countries)]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_countries, x="year", y="co2", hue="country", palette=COUNTRY_COLORS)
    plt.title("CO2 Emissions by Country")
    plt.ylabel("CO2 Emissions (Million Tonnes)")
    plt.xlabel("Year")
    plt.legend(title="Country")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/country_co2_trend.png")
    print(f"Saved {output_dir}/country_co2_trend.png")

    # 3. Korelasyon Analizi
    print("\n--- Correlation Analysis ---")
    cols_to_corr = ["co2", "gdp", "population", "energy_per_capita", "co2_per_capita", "methane", "nitrous_oxide"]
    cols_to_corr = [c for c in cols_to_corr if c in df_eda.columns]
    df_corr = df_eda[df_eda["year"] > 1990][cols_to_corr].dropna()

    if not df_corr.empty:
        corr_matrix = df_corr.corr()
        print(corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix (Post-1990)")
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        print(f"Saved {output_dir}/correlation_matrix.png")

    # 4. Gelişmiş Analiz ve Tahmin (Multivariate)
    print("\n--- Advanced Analysis & Multivariate Prediction ---")

    # Küresel Tahmin
    df_train_global, future_years, pred_global, model_global, ci_lower_global, ci_upper_global = predict_co2_multivariate(df_eda)

    plt.figure(figsize=(12, 6))
    if df_train_global is not None:
        sns.lineplot(data=df_train_global, x="year", y="co2", label="Historical (2000-2024)", color="black")
        plt.plot(future_years, pred_global, color="red", linestyle="--", label="Multivariate Prediction (2025-2028)")
        plt.fill_between(
            future_years.flatten(), ci_lower_global, ci_upper_global, color="red", alpha=0.2, label="95% Confidence Interval"
        )
        plt.title("Global CO2 Emissions Forecast (Multivariate Model)")
        plt.ylabel("CO2 Emissions (Million Tonnes)")
        plt.xlabel("Year")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(f"{output_dir}/global_forecast_multivariate.png")
        print(f"Saved {output_dir}/global_forecast_multivariate.png")

    # Ülke Bazlı Tahminler
    plt.figure(figsize=(14, 7))
    for country in countries:
        df_train, future_years, preds, model, ci_lower, ci_upper = predict_co2_multivariate(df_eda, country)
        if preds is not None:
            color = COUNTRY_COLORS.get(country, "gray")
            plt.plot(df_train["year"], df_train["co2"], label=f"{country} Historical", color=color, alpha=0.6)
            plt.plot(future_years, preds, linestyle="--", label=f"{country} Prediction", color=color, linewidth=2)
            plt.fill_between(future_years.flatten(), ci_lower, ci_upper, color=color, alpha=0.1)

            last_hist = df_train["co2"].iloc[-1]
            last_pred = preds[-1]
            trend = "Increasing" if last_pred > last_hist else "Decreasing"
            print(f"{country}: Trend is {trend} (2028 Pred: {last_pred:.2f} vs Last Hist: {last_hist:.2f})")

    plt.title("CO2 Emissions Forecast by Country (Multivariate, 2025-2028)")
    plt.ylabel("CO2 Emissions (Million Tonnes)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/country_forecasts_multivariate.png")
    print(f"Saved {output_dir}/country_forecasts_multivariate.png")

    # 5. Sürücü Analizi ve Öneriler
    print("\n--- Driver Analysis & Recommendations ---")
    for country in countries:
        country_data = df_eda[df_eda["country"] == country].dropna(subset=["co2", "gdp", "energy_per_capita", "population"])
        if len(country_data) > 10:
            corr = country_data[["co2", "gdp", "energy_per_capita", "population"]].corr()["co2"]
            print(f"\nReport for {country}:")
            print(f"  - CO2 Correlation with GDP: {corr.get('gdp', 0):.2f}")
            print(f"  - CO2 Correlation with Energy: {corr.get('energy_per_capita', 0):.2f}")

            if corr.get("gdp", 0) > 0.9:
                print("  -> Recommendation: Focus on decoupling economic growth from emissions (Green Growth).")
            if corr.get("energy_per_capita", 0) > 0.9:
                print("  -> Recommendation: High energy dependency. Prioritize renewable energy transition and efficiency.")
            if corr.get("population", 0) > 0.9:
                print("  -> Recommendation: Population growth is a major driver. Focus on sustainable urban planning.")

    # 6. Azaltım Senaryoları
    print("\n--- Reduction Scenarios ---")
    target_year = 2050
    current_year = 2024
    years_remaining = target_year - current_year

    for country in countries:
        country_data = df_eda[(df_eda["country"] == country) & (df_eda["year"] == current_year)]
        if not country_data.empty:
            current_co2 = country_data["co2"].values[0]
        else:
            country_data_all = df_eda[df_eda["country"] == country].dropna(subset=["co2"])
            if not country_data_all.empty:
                current_co2 = country_data_all["co2"].iloc[-1]
            else:
                continue

        target_co2 = current_co2 * 0.5
        required_reduction = (1 - (target_co2 / current_co2) ** (1 / years_remaining)) * 100
        print(f"{country}: To halve emissions by 2050, needs {required_reduction:.2f}% annual reduction.")

    # 7. Kişi Başına CO2 Analizi
    print("\n--- CO2 per Capita Analysis ---")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_countries, x="year", y="co2_per_capita", hue="country", palette=COUNTRY_COLORS)
    plt.title("CO2 Emissions per Capita by Country")
    plt.ylabel("CO2 per Capita (Tonnes)")
    plt.xlabel("Year")
    plt.legend(title="Country")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/co2_per_capita_trend.png")
    print(f"Saved {output_dir}/co2_per_capita_trend.png")

    # 8. Nüfus ve CO2 Büyüme Analizi
    print("\n--- Population vs CO2 Growth Analysis ---")
    start_year_growth = 2004
    end_year_growth = 2024

    for country in countries:
        country_data = df_eda[
            (df_eda["country"] == country) & (df_eda["year"] >= start_year_growth) & (df_eda["year"] <= end_year_growth)
        ].sort_values("year")

        if not country_data.empty:
            base_pop = country_data["population"].iloc[0]
            base_co2 = country_data["co2"].iloc[0]

            if base_pop > 0 and base_co2 > 0:
                country_data["pop_index"] = (country_data["population"] / base_pop) * 100
                country_data["co2_index"] = (country_data["co2"] / base_co2) * 100

                fig, ax1 = plt.subplots(figsize=(10, 6))
                color_co2 = COUNTRY_COLORS.get(country, "tab:red")
                color_pop = "black"

                ax1.set_xlabel("Year")
                ax1.set_ylabel("CO2 Emissions (Index 2004=100)", color=color_co2)
                ax1.plot(country_data["year"], country_data["co2_index"], color=color_co2, label="CO2 Growth", linewidth=2)
                ax1.tick_params(axis="y", labelcolor=color_co2)

                ax2 = ax1.twinx()
                ax2.set_ylabel("Population (Index 2004=100)", color=color_pop)
                ax2.plot(country_data["year"], country_data["pop_index"], color=color_pop, linestyle="--", label="Population Growth")
                ax2.tick_params(axis="y", labelcolor=color_pop)

                plt.title(f"{country}: Population vs CO2 Growth (Indexed)")
                fig.tight_layout()
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.savefig(f"{output_dir}/pop_vs_co2_{country}.png")
                print(f"Saved {output_dir}/pop_vs_co2_{country}.png")
                plt.close()

    # 9. Nüfusa Göre Kişi Başına Değişim
    print("\n--- Per Capita Change relative to Population ---")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_countries[df_countries["year"] > 1990], x="population", y="co2_per_capita", hue="country", palette=COUNTRY_COLORS
    )
    plt.title("Population vs CO2 per Capita (Post-1990)")
    plt.ylabel("CO2 per Capita (Tonnes)")
    plt.xlabel("Population")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/population_vs_per_capita.png")
    print(f"Saved {output_dir}/population_vs_per_capita.png")

    # 10. Fosil Yakıt Kaynakları Analizi
    print("\n--- Fossil Fuel Sources Analysis ---")
    fuel_cols = ["coal_co2", "oil_co2", "gas_co2"]
    existing_fuel_cols = [c for c in fuel_cols if c in df_eda.columns]

    if existing_fuel_cols:
        plt.figure(figsize=(12, 8))

        last_year_data = []
        years_used = []

        for country in countries:
            country_df = df_eda[df_eda["country"] == country].sort_values("year")
            if not country_df.empty:
                valid_row = country_df.dropna(subset=existing_fuel_cols).tail(1)
                if not valid_row.empty:
                    last_year_data.append(valid_row)
                    years_used.append(valid_row["year"].iloc[0])

        if last_year_data:
            df_fuel = pd.concat(last_year_data)
            unique_years = sorted(list(set(years_used)))
            year_label = f"{min(unique_years)}-{max(unique_years)}" if len(unique_years) > 1 else str(unique_years[0])

            df_fuel["total_fossil"] = df_fuel[existing_fuel_cols].sum(axis=1)
            for col in existing_fuel_cols:
                df_fuel[f"{col}_share"] = (df_fuel[col] / df_fuel["total_fossil"]) * 100

            plot_data = df_fuel.set_index("country")[[f"{c}_share" for c in existing_fuel_cols]]
            plot_data.columns = [c.replace("_co2", "").title() for c in existing_fuel_cols]
            plot_data.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="viridis")

            plt.title(f"Fossil Fuel CO2 Emission Mix (Year Used: {year_label})")
            plt.ylabel("Percentage Share (%)")
            plt.xlabel("Country")
            plt.legend(title="Fuel Source", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.savefig(f"{output_dir}/fossil_fuel_mix.png")
            print(f"Saved {output_dir}/fossil_fuel_mix.png")
        else:
            print("No valid data found for fossil fuel analysis.")

    # 11. Nüfus Tahmini (2025-2028)
    print("\n--- Population Forecast (2025-2028) ---")
    plt.figure(figsize=(12, 6))
    future_years_pop = np.arange(2025, 2029)

    for country in countries:
        country_data = df_eda[df_eda["country"] == country].dropna(subset=["population"])
        if len(country_data) > 5:
            X_pop = country_data[["year"]]
            y_pop = country_data["population"]

            poly_pop = PolynomialFeatures(degree=2)
            X_poly_pop = poly_pop.fit_transform(X_pop)

            model_pop = LinearRegression()
            model_pop.fit(X_poly_pop, y_pop)

            future_years_reshaped = future_years_pop.reshape(-1, 1)
            future_poly_pop = poly_pop.transform(future_years_reshaped)
            pred_pop = model_pop.predict(future_poly_pop)

            color = COUNTRY_COLORS.get(country, "gray")
            plt.plot(country_data["year"], country_data["population"] / 1e6, label=f"{country} (Hist)", color=color)
            plt.plot(future_years_pop, pred_pop / 1e6, linestyle="--", color=color)

    plt.title("Population Forecast (2025-2028)")
    plt.ylabel("Population (Millions)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/population_forecast.png")
    print(f"Saved {output_dir}/population_forecast.png")

    # 12. CO2 Etki Analizi (Population Driven)
    print("\n--- CO2 Impact Analysis (Population Driven) ---")
    plt.figure(figsize=(12, 6))

    for country in countries:
        country_data = df_eda[df_eda["country"] == country].dropna(subset=["co2", "population", "co2_per_capita"])
        if not country_data.empty:
            last_hist_year = country_data["year"].max()
            last_per_capita = country_data.loc[country_data["year"] == last_hist_year, "co2_per_capita"].values[0]

            X_pop = country_data[["year"]]
            y_pop = country_data["population"]
            poly_pop = PolynomialFeatures(degree=2)
            X_poly_pop = poly_pop.fit_transform(X_pop)
            model_pop = LinearRegression()
            model_pop.fit(X_poly_pop, y_pop)

            future_years_reshaped = future_years_pop.reshape(-1, 1)
            future_poly_pop = poly_pop.transform(future_years_reshaped)
            pred_pop = model_pop.predict(future_poly_pop)

            pop_driven_co2 = pred_pop * last_per_capita
            color = COUNTRY_COLORS.get(country, "gray")
            plt.plot(future_years_pop, pop_driven_co2, linestyle=":", linewidth=2, color=color, label=f"{country} (Pop. Driven)")

    plt.title("Projected CO2 if Per Capita Emissions Remain Constant (2025-2028)")
    plt.ylabel("CO2 Emissions (Million Tonnes)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/co2_impact_analysis.png")
    print(f"Saved {output_dir}/co2_impact_analysis.png")

    # 13. Üretim vs Tüketim Bazlı Emisyon Analizi
    print("\n--- Production vs Consumption Analysis ---")
    for country in countries:
        country_data = df_eda[df_eda["country"] == country].sort_values("year")

        if "consumption_co2" in country_data.columns and not country_data["consumption_co2"].isnull().all():
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=country_data,
                x="year",
                y="co2",
                label="Production (Territorial)",
                color=COUNTRY_COLORS.get(country, "blue"),
                linewidth=2,
            )
            sns.lineplot(
                data=country_data,
                x="year",
                y="consumption_co2",
                label="Consumption (Trade-Adjusted)",
                color="gray",
                linestyle="--",
                linewidth=2,
            )
            plt.fill_between(country_data["year"], country_data["co2"], country_data["consumption_co2"], alpha=0.1, color="gray")
            plt.title(f"{country}: Production vs Consumption CO2 Emissions")
            plt.ylabel("CO2 Emissions (Million Tonnes)")
            plt.xlabel("Year")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            filename = f"{output_dir}/prod_vs_cons_{country}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()
        else:
            print(f"Skipping {country}: No consumption data.")

    # 14. Ekonomik Karbon Yoğunluğu (CO2 per GDP)
    print("\n--- Carbon Intensity Analysis (CO2 per GDP) ---")
    plt.figure(figsize=(12, 7))
    for country in countries:
        country_data = df_eda[df_eda["country"] == country].sort_values("year")
        country_data = country_data[country_data["year"] >= 2000]
        if "co2_per_gdp" in country_data.columns:
            sns.lineplot(
                data=country_data,
                x="year",
                y="co2_per_gdp",
                label=country,
                color=COUNTRY_COLORS.get(country),
                linewidth=2,
            )
    plt.title("Economic Carbon Intensity (CO2 per GDP) Trend (2000-2024)")
    plt.ylabel("CO2 per GDP (kg per $)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/carbon_intensity_trend.png")
    print(f"Saved {output_dir}/carbon_intensity_trend.png")



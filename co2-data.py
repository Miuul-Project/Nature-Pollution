import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
from datetime import date
import os
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)


def load():
    try:
        data = pd.read_csv("Datasets/owid-co2-data.csv")
    except FileNotFoundError:
        # Fallback if running from parent directory
        data = pd.read_csv("Nature-Pollution-main/Datasets/owid-co2-data.csv")
    return data

df = load()

def clean_and_balance_data(data):
    print("\n--- Data Quality Report (Before Cleaning) ---")
    print("Missing Values (%):")
    print(data[['co2', 'population', 'gdp']].isnull().mean() * 100)
    
    # Check data count for key countries
    countries_check = ['China', 'United States', 'Russia', 'Turkey', 'Germany']
    print("\nData Points per Country (Selected):")
    print(data[data['country'].isin(countries_check)]['country'].value_counts())

    # Interpolate missing values within each country
    # Sort by country and year to ensure correct interpolation
    data = data.sort_values(['country', 'year'])
    
    # Group by country and interpolate numeric columns
    # We use forward fill and backward fill as a fallback for edges if interpolation leaves NaNs
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Using apply to handle the group by transformation safely
    def fill_group(group):
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear', limit_direction='both')
        return group

    data = data.groupby('country', group_keys=False).apply(fill_group)
    
    print("\n--- Data Quality Report (After Interpolation) ---")
    print("Missing Values (%):")
    print(data[['co2', 'population', 'gdp']].isnull().mean() * 100)
    
    return data

df = clean_and_balance_data(df)

# Ensure img directory exists
output_dir = "img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define Standard Colors
COUNTRY_COLORS = {
    'China': 'tab:red',
    'United States': 'tab:blue',
    'Germany': 'gold', # Yellow can be hard to see, gold is better
    'Russia': 'tab:purple',
    'Turkey': 'tab:orange'
}

# 1. General CO2 Increase Over Years
print("--- General CO2 Increase Over Years ---")
yearly_co2 = df.groupby('year')['co2'].mean()
print(yearly_co2.tail())

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='year', y='co2', estimator='mean', errorbar=None, color='black') # Global trend in black
plt.title('Average Global CO2 Emissions Over Years')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/global_co2_trend.png')
print(f"Saved {output_dir}/global_co2_trend.png")

# 2. Country-Specific Analysis
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

# 3. Correlation Analysis
print("\n--- Correlation Analysis ---")
# Select numerical columns relevant to CO2 and development
cols_to_corr = ['co2', 'gdp', 'population', 'energy_per_capita', 'co2_per_capita']
# Filter for recent years to get more relevant data and drop NaNs for correlation
df_corr = df[df['year'] > 1990][cols_to_corr].dropna()

corr_matrix = df_corr.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Post-1990)')
plt.savefig(f'{output_dir}/correlation_matrix.png')
print(f"Saved {output_dir}/correlation_matrix.png")

# 4. Advanced Analysis & Prediction
print("\n--- Advanced Analysis & Prediction ---")

def predict_co2(data, country_name=None):
    # Filter data for the specific country if provided, else use global data
    if country_name:
        df_subset = data[data['country'] == country_name]
        title_suffix = f" ({country_name})"
    else:
        df_subset = data.groupby('year')['co2'].mean().reset_index()
        title_suffix = " (Global Average)"
    
    # Use data from 2000 to 2024 for training to capture recent trends
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)].dropna(subset=['co2'])
    
    if len(df_train) < 5:
        print(f"Not enough data for {title_suffix}")
        return None, None, None, None # Fixed unpacking error

    X = df_train[['year']]
    y = df_train['co2']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for next 6 years (2025-2030)
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return df_train, future_years, predictions, model

# Global Prediction
df_train_global, future_years, pred_global, model_global = predict_co2(df)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train_global, x='year', y='co2', label='Historical (2000-2024)', color='black')
plt.plot(future_years, pred_global, color='red', linestyle='--', label='Prediction (2025-2030)')
plt.title('Global CO2 Emissions Forecast')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/global_forecast.png')
print(f"Saved {output_dir}/global_forecast.png")

# Country Predictions
plt.figure(figsize=(14, 7))
for country in countries:
    df_train, future_years, preds, model = predict_co2(df, country)
    if preds is not None:
        # Plotting historical tail and prediction
        color = COUNTRY_COLORS.get(country, 'gray')
        plt.plot(df_train['year'], df_train['co2'], label=f'{country} Historical', color=color, alpha=0.6)
        plt.plot(future_years, preds, linestyle='--', label=f'{country} Prediction', color=color, linewidth=2)
        
        # Trend Analysis
        trend = "Increasing" if model.coef_[0] > 0 else "Decreasing"
        print(f"{country}: Trend is {trend} (Slope: {model.coef_[0]:.2f})")

plt.title('CO2 Emissions Forecast by Country (2025-2030)')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/country_forecasts.png')
print(f"Saved {output_dir}/country_forecasts.png")

# 5. Driver Analysis & Recommendations
print("\n--- Driver Analysis & Recommendations ---")
# Analyze what correlates most with CO2 for each country to give specific advice
for country in countries:
    country_data = df[df['country'] == country].dropna(subset=['co2', 'gdp', 'energy_per_capita', 'population'])
    if len(country_data) > 10:
        # Simple correlation to find drivers
        corr = country_data[['co2', 'gdp', 'energy_per_capita', 'population']].corr()['co2']
        
        print(f"\nReport for {country}:")
        print(f"  - CO2 Correlation with GDP: {corr['gdp']:.2f}")
        print(f"  - CO2 Correlation with Energy: {corr['energy_per_capita']:.2f}")
        
        # Recommendations
        if corr['gdp'] > 0.9:
            print("  -> Recommendation: Focus on decoupling economic growth from emissions (Green Growth).")
        if corr['energy_per_capita'] > 0.9:
            print("  -> Recommendation: High energy dependency. Prioritize renewable energy transition and efficiency.")
        if corr['population'] > 0.9:
            print("  -> Recommendation: Population growth is a major driver. Focus on sustainable urban planning.")
            
# 6. Reduction Scenarios
print("\n--- Reduction Scenarios ---")
# Calculate required reduction to reach 50% of 2024 levels by 2050
target_year = 2050
current_year = 2024
years_remaining = target_year - current_year

for country in countries:
    country_data = df[(df['country'] == country) & (df['year'] == current_year)]
    if not country_data.empty:
        current_co2 = country_data['co2'].values[0]
        target_co2 = current_co2 * 0.5
        
        # Compound Annual Reduction Rate
        required_reduction = (1 - (target_co2 / current_co2) ** (1 / years_remaining)) * 100
        print(f"{country}: To halve emissions by 2050, needs {required_reduction:.2f}% annual reduction.")

# 7. CO2 per Capita Analysis
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

# 8. Population vs CO2 Growth Analysis
print("\n--- Population vs CO2 Growth Analysis ---")
# Calculate growth rates for the last 20 years (approx 2004-2024)
start_year_growth = 2004
end_year_growth = 2024

for country in countries:
    country_data = df[(df['country'] == country) & (df['year'] >= start_year_growth) & (df['year'] <= end_year_growth)].sort_values('year')
    
    if not country_data.empty:
        # Normalize to start year = 100 for comparison
        base_pop = country_data['population'].iloc[0]
        base_co2 = country_data['co2'].iloc[0]
        
        country_data['pop_index'] = (country_data['population'] / base_pop) * 100
        country_data['co2_index'] = (country_data['co2'] / base_co2) * 100
        
        # Dual Axis Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Use country color for CO2, and a neutral color (e.g., grey/black) for Population to contrast
        # Or use the requested specific colors? 
        # User said: "Ülke renkleri her grafik için sabit olmalı". 
        # But here we have 2 lines for 1 country. 
        # Let's use the Country Color for CO2 (the main subject) and a standard color for Population (e.g., Black dashed)
        
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

# 9. Per Capita Change relative to Population
print("\n--- Per Capita Change relative to Population ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_countries[df_countries['year'] > 1990], x='population', y='co2_per_capita', hue='country', palette=COUNTRY_COLORS)
plt.title('Population vs CO2 per Capita (Post-1990)')
plt.ylabel('CO2 per Capita (Tonnes)')
plt.xlabel('Population')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/population_vs_per_capita.png')
print(f"Saved {output_dir}/population_vs_per_capita.png")

# 10. Population Prediction & CO2 Impact Analysis
print("\n--- Population Prediction & CO2 Impact Analysis ---")

def predict_population(data, country_name):
    df_subset = data[data['country'] == country_name].dropna(subset=['population'])
    # Use data from 2000 to 2024 for training
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)]
    
    if len(df_train) < 5:
        return None, None, None

    X = df_train[['year']]
    y = df_train['population']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for next 10 years (2025-2035)
    future_years = np.arange(2025, 2036).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return df_train, future_years, predictions

def predict_co2_from_population(data, country_name, future_pop):
    df_subset = data[data['country'] == country_name].dropna(subset=['population', 'co2'])
    # Use data from 2000 to 2024
    df_train = df_subset[(df_subset['year'] >= 2000) & (df_subset['year'] <= 2024)]
    
    if len(df_train) < 5:
        return None
        
    X = df_train[['population']]
    y = df_train['co2']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict CO2 based on future population
    # future_pop is a numpy array from the previous step
    future_co2 = model.predict(future_pop.reshape(-1, 1))
    
    return future_co2

# Visualization: Population Forecast
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

# Visualization: CO2 Impact Analysis (Population-Driven vs Time-Series Forecast)
# We will compare the CO2 prediction based purely on Population growth vs the Time-based prediction
plt.figure(figsize=(14, 7))

for country in countries:
    # 1. Get Population Forecast
    _, future_years, pop_preds = predict_population(df, country)
    
    if pop_preds is not None:
        # 2. Predict CO2 based on Population Forecast
        co2_from_pop = predict_co2_from_population(df, country, pop_preds)
        
        if co2_from_pop is not None:
            # Plot only the forecast part for clarity, or overlay
            # Let's plot the "Impact" line (CO2 if it follows population trend)
            color = COUNTRY_COLORS.get(country, 'gray')
            plt.plot(future_years, co2_from_pop, linestyle=':', marker='.', label=f'{country} (Pop-Driven CO2)', color=color)

plt.title('Projected CO2 Emissions based ONLY on Population Growth (2025-2035)')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/co2_impact_analysis.png')
print(f"Saved {output_dir}/co2_impact_analysis.png")

# 11. Fossil Fuel Sources Analysis
print("\n--- Fossil Fuel Sources Analysis ---")
# Analyze the share of Coal, Oil, and Gas in CO2 emissions for the most recent available year
fuel_cols = ['coal_co2', 'oil_co2', 'gas_co2']
# Check if columns exist
existing_fuel_cols = [c for c in fuel_cols if c in df.columns]

if existing_fuel_cols:
    plt.figure(figsize=(12, 8))
    
    # Get data for the last year for each country
    last_year_data = []
    
    for country in countries:
        country_df = df[df['country'] == country].sort_values('year')
        if not country_df.empty:
            # Get the last row with valid data for these columns
            valid_row = country_df.dropna(subset=existing_fuel_cols).tail(1)
            if not valid_row.empty:
                last_year_data.append(valid_row)
    
    if last_year_data:
        df_fuel = pd.concat(last_year_data)
        
        # Calculate percentage share
        # Note: These columns are usually absolute values (tonnes). We need to sum them to get total fossil CO2 (or use total co2 if we want share of total)
        # Let's calculate share of the sum of these three sources to show the "Mix"
        df_fuel['total_fossil'] = df_fuel[existing_fuel_cols].sum(axis=1)
        
        for col in existing_fuel_cols:
            df_fuel[f'{col}_share'] = (df_fuel[col] / df_fuel['total_fossil']) * 100
            
        # Plotting Stacked Bar Chart
        plot_data = df_fuel.set_index('country')[[f'{c}_share' for c in existing_fuel_cols]]
        plot_data.columns = [c.replace('_co2', '').title() for c in existing_fuel_cols] # Rename for legend
        
        plot_data.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
        
        plt.title(f'Fossil Fuel CO2 Emission Mix (Most Recent Data)')
        plt.ylabel('Percentage Share (%)')
        plt.xlabel('Country')
        plt.legend(title='Fuel Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/fossil_fuel_mix.png')
        print(f"Saved {output_dir}/fossil_fuel_mix.png")
        
        # Print the specific values for the report
        print("\nFossil Fuel Mix Data:")
        print(plot_data)
    else:
        print("No valid data found for fossil fuel analysis.")
else:
    print("Fossil fuel columns not found in dataset.")

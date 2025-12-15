![License](https://img.shields.io/badge/license-Educational-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![Language](https://img.shields.io/badge/language-Python-yellow.svg)

# 🌍 Global CO₂ Analysis & Future Projections

## 👥 Project Members

| Name | LinkedIn | GitHub | Website |
|------|----------|--------|---------|
| **Alican Kaya** | [LinkedIn](https://www.linkedin.com/in/alican-kaya-881650234/) | [GitHub](https://github.com/AlicanKaya192) | [Website](https://alican-kaya.com/) |
| **Sude Şenol** | [LinkedIn](http://www.linkedin.com/in/sude-%C5%9Fenol) | [GitHub](https://github.com/sudesenoll) | — |
| **Zülal Özge** | [LinkedIn](https://www.linkedin.com/in/z%C3%BClal-%C3%B6zge-687488333) | [GitHub](https://github.com/zulalozge) | — |
| **Hasret Erdoğan** | [LinkedIn](https://www.linkedin.com/in/hasret-erdoğan-5b463b278) | [GitHub](https://github.com/hasreterdogan) | — |
| **Duru Bağdadioğlu** | [LinkedIn](https://www.linkedin.com/in/duru-ba%C4%9Fdadio%C4%9Flu/) | [GitHub](https://github.com/durubagdadioglu) | — |

## 📖 Project Overview
This project presents a comprehensive data analysis of global CO₂ emission trends, aiming to uncover the root causes of climate change beyond simple emission statistics. By leveraging historical data from the Industrial Revolution to the present day, we explore the complex relationships between economic growth, population dynamics, energy consumption, and environmental impact.

The study goes beyond visualization to provide **machine learning-based forecasts** for 2025-2028 and strategic recommendations for key global players (China, USA, India, etc.).

## 📊 Data Story
Our analysis is built upon the **Our World in Data** CO₂ dataset, a gold-standard resource in climate science.
*   **Scope:** From the Industrial Revolution to 2024.
*   **Focus:** We concentrate on the modern era (post-1990) to understand current geopolitical and environmental trends.
*   **Philosophy:** We don't just ask *"How much are we polluting?"* but *"Why are we polluting?"*. To answer this, we integrate variables like **GDP, Population, and Energy Mix** into our models.

## 🔍 Key Analysis Modules

### 1. Historical Trends & Country Profiles
*   Tracking the rise of global emissions.
*   Comparative analysis of major economies: **China, USA, India, Russia, Germany, Turkey**.

### 2. Drivers of Pollution (Correlation Analysis)
*   Identifying the strongest predictors of CO₂ emissions (e.g., Population vs. GDP).
*   Heatmaps revealing the "engine" of carbon output.

### 3. Future Projections (ML Forecasting)
*   **Multivariate Polynomial Regression** models trained on 2000-2024 data.
*   Forecasting emissions through 2028 based on projected population and economic growth.

### 4. Advanced Metrics
*   **Production vs. Consumption:** Analyzing "Carbon Leakage"—do developed nations outsource their pollution?
*   **Carbon Intensity (CO₂/GDP):** Measuring the "greenness" of economic growth.
*   **Fossil Fuel Dependency:** Breaking down emissions by source (Coal, Oil, Gas).

## 🛠️ Technologies & Methodology
*   **Python:** Core programming language.
*   **Pandas:** Advanced data manipulation and cleaning (Linear Interpolation for missing values).
*   **Scikit-learn:** Machine Learning for regression models and forecasting.
*   **Matplotlib & Seaborn:** Professional-grade data visualization.

## 📂 Project Structure
*   `co2-data.py`: The main analytical engine. Performs data cleaning, modeling, and generates all visualizations.
*   `Datasets/`: Contains the raw `owid-co2-data.csv`.
*   `img/`: Stores the generated charts and graphs used in reports.
*   `metrics.json`: Stores model performance metrics (RMSE, R², MAE).
*   `*.pdf`: Generated analysis reports and presentation guides.

---
*This project is part of the Miuul Data Science Bootcamp.*

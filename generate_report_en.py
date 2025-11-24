from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CO2 Data Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

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
            self.image(image_path, w=170)
            self.ln(2)
            if title:
                self.set_font('Arial', 'I', 10)
                self.cell(0, 5, title, 0, 1, 'C')
            self.ln(5)
        else:
            self.cell(0, 10, f"Image not found: {image_path}", 0, 1)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Content - Professional & Objective Tone
pdf.chapter_body("This report presents a comprehensive analysis of CO2 emission trends, comparing key nations and exploring the underlying factors driving these changes. The study aims to provide data-driven insights into global emission patterns and future trajectories.")

pdf.chapter_title("1. Global CO2 Trends: Historical Overview")
pdf.chapter_body("Historical data indicates a consistent upward trajectory in global average CO2 emissions over the examined period. This trend underscores the growing challenge of atmospheric carbon accumulation.")
pdf.add_image("img/global_co2_trend.png", "Global CO2 Trend")

pdf.chapter_title("2. Country-Specific Observations")
pdf.chapter_body("A comparative analysis of China, USA, Russia, Turkey, and Germany reveals distinct emission profiles:\n"
                    "- China: Data shows a significant surge in emissions over recent decades, correlating with rapid industrialization.\n"
                    "- USA: While historically high, recent data indicates a slight declining trend, suggesting a shift in energy policy or efficiency.\n"
                    "- Germany & Russia: Emissions appear relatively stable or show a slight decline.\n"
                    "- Turkey: A gradual but consistent increase in emissions is observed.")
pdf.add_image("img/country_co2_trend.png", "Country-Specific CO2 Trend")

pdf.chapter_title("3. Correlation Analysis Results")
pdf.chapter_body("Statistical analysis of post-1990 data highlights key drivers:\n"
                    "- A strong positive correlation exists between CO2 emissions, GDP, and Population.\n"
                    "- Energy per Capita shows a significant relationship with CO2 per Capita, reinforcing the link between energy consumption patterns and individual carbon footprints.")
pdf.add_image("img/correlation_matrix.png", "Correlation Matrix")

pdf.chapter_title("4. Forecast Projections (2025-2030)")
pdf.chapter_body("Projections based on a Linear Regression model trained on 2000-2024 data suggest:\n"
                    "- Global emissions are likely to continue their upward trend absent significant intervention.")
pdf.add_image("img/global_forecast.png", "Global Forecast")

pdf.chapter_title("5. Country-Specific Forecasts")
pdf.chapter_body("Model projections for key nations indicate:\n"
                    "- China & Turkey: Projected to maintain an increasing trend.\n"
                    "- USA & Germany: Projected to continue their declining trend.\n"
                    "- Russia: Projected to remain stable with a slight potential increase.")
pdf.add_image("img/country_forecasts.png", "Country Forecasts")

pdf.chapter_title("6. Recommendations & Reduction Scenarios")
pdf.chapter_body("To achieve a target of halving emissions by 2050 (relative to 2024 levels), the following annual reductions are required:\n\n"
                    "China (~2.63% Annual Reduction): Driven by GDP. Recommendation: Focus on decoupling economic growth from emissions (Green Growth strategies).\n"
                    "USA (~2.63%): Driven by GDP. Recommendation: Continue decoupling efforts and prioritize energy efficiency.\n"
                    "Turkey (~2.63%): Driven by Energy. Recommendation: Address high energy dependency by prioritizing renewable energy transition.\n"
                    "Germany (~2.63%): Driven by GDP. Recommendation: Maintain and expand current green growth initiatives.\n"
                    "Russia (~2.63%): Driven by Energy. Recommendation: Strategic shift away from fossil fuel dependency is advised.")

pdf.chapter_title("7. CO2 per Capita Analysis")
pdf.chapter_body("Analysis of per capita emissions provides insight into emission intensity relative to population:\n"
                    "- The USA maintains the highest per capita emissions, though a downward trend is evident.\n"
                    "- China's per capita emissions have risen significantly but remain below US levels.")
pdf.add_image("img/co2_per_capita_trend.png", "CO2 per Capita Trend")

pdf.chapter_title("8. Population vs. CO2 Growth Dynamics")
pdf.chapter_body("Comparing population growth with CO2 emissions (indexed to 2004) reveals distinct divergence patterns across different nations. This analysis highlights the effectiveness of decoupling strategies.\n\n"
                    "China: CO2 emissions have outpaced population growth significantly, reflecting rapid industrial expansion.")
pdf.add_image("img/pop_vs_co2_China.png", "China: Population vs CO2")

pdf.chapter_body("USA: Emissions have declined despite population growth, indicating successful decoupling of demographics from carbon output through efficiency gains and policy shifts.")
pdf.add_image("img/pop_vs_co2_United States.png", "USA: Population vs CO2")

pdf.chapter_body("Germany: Similar to the USA, Germany exhibits a strong decoupling trend, where emissions fall while the population remains relatively stable.")
pdf.add_image("img/pop_vs_co2_Germany.png", "Germany: Population vs CO2")

pdf.chapter_body("Russia: The data shows a complex relationship where emissions and population trends are less divergent compared to Western economies, often fluctuating with economic conditions.")
pdf.add_image("img/pop_vs_co2_Russia.png", "Russia: Population vs CO2")

pdf.chapter_body("Turkey: Emissions are growing in tandem with or faster than the population, suggesting that economic growth is still heavily carbon-intensive.")
pdf.add_image("img/pop_vs_co2_Turkey.png", "Turkey: Population vs CO2")

pdf.chapter_title("9. Demographic Drivers: Population vs Per Capita")
pdf.chapter_body("The relationship between population size and per capita emissions is non-linear. This suggests that development stage and energy policy, rather than population size alone, are the primary determinants of per capita emissions.")
pdf.add_image("img/population_vs_per_capita.png", "Population vs Per Capita")

pdf.chapter_title("10. Population Forecast (2025-2035)")
pdf.chapter_body("Demographic projections for the coming decade indicate:\n"
                    "- China: Population is projected to peak and potentially begin a decline.\n"
                    "- USA & Turkey: Continued population growth is anticipated.\n"
                    "- Russia & Germany: Population levels are expected to remain stable or decline slightly.")
pdf.add_image("img/population_forecast.png", "Population Forecast")

pdf.chapter_title("11. Impact Analysis: Population-Driven Emissions")
pdf.chapter_body("Modeling the isolated impact of population growth on CO2 emissions reveals:\n"
                    "- This projection assumes historical emission-per-capita trends remain constant.\n"
                    "- Divergence Analysis: Where the 'Population-Driven' projection exceeds the 'Actual Forecast', it indicates that the country is successfully improving efficiency and decoupling emissions from population growth.")
pdf.add_image("img/co2_impact_analysis.png", "CO2 Impact Analysis")

pdf.chapter_title("12. Projected Consequences of Unchecked Growth")
pdf.chapter_body("The concurrent escalation of atmospheric CO2 concentrations and global population density poses severe systemic risks. Scientific analysis identifies three primary domains of impact:\n\n"
                    "1. Environmental Destabilization: Elevated CO2 levels correlate directly with rising global mean temperatures, leading to increased frequency of extreme weather events, glacial retreat, and ocean acidification. These changes threaten global biodiversity and ecosystem integrity.\n\n"
                    "2. Resource Scarcity: Rapid demographic expansion exerts compounding pressure on finite natural resources. Water scarcity, arable land degradation, and depletion of non-renewable energy reserves are projected to intensify, potentially triggering geopolitical instability.\n\n"
                    "3. Public Health Implications: Deteriorating air quality and changing vector ecology present significant health challenges. Respiratory ailments associated with pollution and the spread of vector-borne diseases due to warming climates represent growing public health burdens.")

pdf.chapter_title("13. Strategic Mitigation Pathways")
pdf.chapter_body("Addressing these challenges requires a multi-dimensional approach, integrating policy, technology, and social adaptation. Analysis suggests the following high-impact strategies:\n\n"
                    "1. Accelerated Decarbonization: Transitioning energy grids from fossil fuels to renewable sources (solar, wind, nuclear) is paramount. This involves not just generation, but also grid modernization and storage solutions.\n\n"
                    "2. Circular Economy Adoption: Shifting from a linear 'take-make-dispose' model to a circular economy can significantly reduce resource extraction and waste generation. This includes prioritizing material recycling, product longevity, and sustainable manufacturing processes.\n\n"
                    "3. Sustainable Urbanization: As population centers grow, implementing smart city infrastructure is critical. This encompasses efficient public transit systems, green building standards, and urban planning that minimizes the carbon footprint per capita.\n\n"
                    "4. Policy & Education: Robust international climate accords, carbon pricing mechanisms, and widespread education on sustainable consumption are essential to drive behavioral change and enforce systemic accountability.")

pdf.chapter_title("14. Fossil Fuel Dependency Analysis")
pdf.chapter_body("To validate the recommendations, we analyzed the composition of CO2 emissions by source (Coal, Oil, Gas). The data confirms distinct energy profiles:\n\n"
                    "- China: Heavily reliant on Coal (>70% of fossil emissions), confirming the need for green growth strategies that target industrial coal usage.\n"
                    "- Russia: Dominated by Gas and Oil (combined >80%), supporting the recommendation to shift away from fossil fuel dependency.\n"
                    "- USA: A mixed profile with significant Oil and Gas usage, reflecting a transport and heating-heavy emission profile.\n"
                    "- Turkey: Shows a significant share of Coal and Gas, necessitating a dual strategy of renewable transition and efficiency.\n"
                    "- Germany: Despite renewables, Coal remains a significant factor, justifying the continued focus on phasing out coal power.")
pdf.add_image("img/fossil_fuel_mix.png", "Fossil Fuel Emission Mix")

pdf.output("CO2_Analysis_Report_Professional.pdf")
print("PDF created: CO2_Analysis_Report_Professional.pdf")

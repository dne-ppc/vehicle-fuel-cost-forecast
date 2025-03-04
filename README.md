# Vehicle Operating Cost Forecast App

This project is an interactive web application (built with [Streamlit](https://streamlit.io/)) that forecasts and compares the net present value (NPV) of operating different vehicle models. It leverages 2016-2025 U.S. Department of Energy data, focusing on **North American gasoline** and **electric** models only. By configuring distributions for parameters like fuel price, annual distance, and interest rate, users can run Monte Carlo simulations to see how costs might vary under different assumptions about the future.

---

## Table of Contents
- [Key Features](#key-features)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
  - [Selecting Models](#selecting-models)
  - [Configuring Distributions](#configuring-distributions)
  - [Running Forecasts](#running-forecasts)
  - [Examining NPV Sensitivity](#examining-npv-sensitivity)
  - [Distribution Analysis](#distribution-analysis)
  - [Process Explanation](#process-explanation)
- [Configuration](#configuration)
- [License](#license)
- [Contact](#contact)

---

## Key Features

1. **Vehicle Selection**  
   - Choose from a curated set of North American **gasoline** and **electric** vehicles (2016 model year).  
   - Filter by vehicle class, year, and quickly identify the best or worst performers by efficiency.

2. **Monte Carlo Simulation**  
   - Configure the number of **iterations** and **years** to simulate.  
   - Incorporate uncertainty in key parameters (e.g., **fuel/electricity costs**, **annual distance**, **interest rate**) via *Metalog distributions*.

3. **NPV Calculation**  
   - Discount future operating costs (fuel/electricity) to present value.  
   - Easily compare the net result of two different vehicle choices (NPV of A minus NPV of B).

4. **Sensitivity Analysis**  
   - Tornado charts reveal which parameters most affect NPV outcomes.  
   - Lock each parameter to its minimum or maximum bound to see the resulting shift in total cost.

5. **Distribution Exploration**  
   - Examine the probability distribution of total NPV (PDF and CDF) for each chosen vehicle.  
   - Understand best-, worst-, and most-likely cost scenarios (via P10, P50, P90 percentiles).

6. **Modular Codebase**  
   - Organized into separate modules (`analysis.py`, `plotting.py`, `layout.py`, etc.) for clarity.  
   - Straightforward to extend or integrate with other data sets or models.

---

## Data Sources

- **Vehicle Model Data**: 2016-2025 U.S. Department of Energy data for North American gasoline and electric vehicles.  
- **Fuel & Electricity Prices**: User-input distributions, referencing typical real-world price ranges.  
- **Interest Rates**: Configurable distributions to approximate discount rate uncertainty.

---

## Project Structure
├── app.py # Main entry point (Streamlit) 
├── analysis.py # Monte Carlo and NPV sensitivity analysis 
├── plotting.py # Plotting utilities (Plotly / Streamlit) 
├── layout.py # Defines each tab / section of the Streamlit UI 
├── controls.py # Handles model selection & distribution inputs 
├── data/
│  ├── models.csv # Vehicle data from US DOE (2016)
│  ├── process_model_eff.ipynb # Jupyter notebook with additional math / references 
│  └── config.yaml # Distributions & settings 
├── requirements.txt # Python dependencies
└── README.md # This file

---

## Installation & Requirements

1. **Clone** this repository:  
   ```bash
   git clone https://github.com/yourusername/vehicle-cost-forecast.git
   cd vehicle-cost-forecast
2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # or .\venv\Scripts\activate on Windows
3. Install dependencies:
    ```bash
    pip install -r requirements.txt

Ensure you have a modern version of Python (e.g., 3.9+). The key packages used include:

- streamlit
- pandas
- plotly
- numpy
- pyyaml
- metalog

---

## Running the App

Once dependencies are installed:
    ```bash
    streamlit run app.py

Your web browser should open automatically. If not, copy the URL displayed in your terminal into a browser (usually http://localhost:8501).

---

## Usage Guide

1. **Select Models**  
   - Navigate to the **Simulation Controls** (often in the sidebar or a dedicated tab).  
   - Choose **one or two** vehicles from the multi-select dropdown.  
   - Specify the number of **years** to forecast and **iterations** for the Monte Carlo simulation.

2. **Configure Distributions**  
   - For each parameter (gasoline price, electricity price, annual distance, interest rate), set the **P10, P50, and P90** values in the UI.  
   - The app uses a [Metalog distribution](https://rdrr.io/cran/metalog/f/vignettes/metalog.Rmd) to build flexible, bounded probability distributions from your inputs.

3. **Run Forecasts**  
   - In the **Forecasts** tab, choose a cost measure (`dollars_per_km`, `dollars`, or `npv`) to plot over time.  
   - You’ll see percentile lines (P10, P50, P90) for each selected scenario (City, Combined, Highway).

4. **Analyze NPV Sensitivity**  
   - The **NPV Sensitivity** tab shows a **tornado chart**.  
   - For each parameter, the app recalculates NPV when that parameter is locked to its min or max value—highlighting the biggest cost drivers.

5. **Check NPV Distribution**  
   - The **NPV Distribution** tab displays the **PDF & CDF** of your total NPV.  
   - See the spread of possible outcomes by examining P10, P50, P90, or other percentiles.

6. **Review the Process**  
   - The **Process Explanation** tab details how MPG is converted to kWh/km, how distributions are sampled, and how the final NPV is calculated.

---

## Configuration
All general settings for your distributions are specified in data/config.yaml. For each entry, you’ll find:

- label: Display label in the UI.
- min_value / max_value: Bounds for the parameter.
- p10 / p50 / p90: Default percentile points in the Metalog distribution.

You can extend or modify these entries as needed (e.g., to add new fuel types, interest rate models, or brand-new parameters). If you add new parameters you will need to update the calculation in analysis.monte_carlo_forecast 

---

## License
This project is released under the MIT License (or whichever license you choose). Refer to the license file for more details.

---

## Contact
For questions, suggestions, or collaborations, feel free to reach out to:

David Nelson Elske (david@peopleandplanet.consulting)

Or open an issue on GitHub if you find a bug or have a feature request!
If you find this tool useful, consider sharing it with colleagues or giving the repository a star on GitHub.
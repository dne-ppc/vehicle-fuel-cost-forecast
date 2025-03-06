from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np

# We'll import your existing modules:
import data
import analysis
import controls

app = FastAPI(
    title="EV Savings API",
    description="A simple FastAPI integration for the EV-savings project.",
    version="0.1.0"
)

# ------------------------------------------------------------------------------
# 1) GET /models : Return the list of available vehicle models
# ------------------------------------------------------------------------------

@app.get("/models")
def list_models():
    """
    Returns a list of all available models in the data.models DataFrame.
    Example response:
      [
        {
          "ID": "Model123",
          "Fuel": "Gasoline",
          "Year": 2020,
          "Veh Class": "Sedan",
          "City": 0.2,
          "Combined": 0.18,
          "Highway": 0.16
        },
        ...
      ]
    """
    # Convert data.models (a DataFrame) to a list of dictionaries
    models_list = data.models.to_dict(orient="records")
    return models_list


# ------------------------------------------------------------------------------
# 2) POST /forecast : Run the forecast with user inputs, return model results
# ------------------------------------------------------------------------------

# Pydantic Models for request body
class DistributionSettings(BaseModel):
    """
    A simplified schema for distribution bounds or values.
    You can expand this or replicate the triple (p10, p50, p90) approach.
    """
    p10: float
    p50: float
    p90: float

class ForecastRequest(BaseModel):
    """
    The user sends:
      - selected_models: list of vehicle IDs
      - years: number of forecast years
      - iterations: number of Monte Carlo draws
      - distributions: a dictionary of distribution settings for each parameter
        (like fuel price, annual distance, etc.)
    Example:
    {
      "selected_models": ["ToyotaCamry2020", "HondaCivic2021"],
      "years": 10,
      "iterations": 1000,
      "distributions": {
        "Annual Distance (km)": {"p10": 10000, "p50": 15000, "p90": 20000},
        "Price of Gasoline ($/L)": {"p10": 0.8, "p50": 1.2, "p90": 1.6},
        "Price of Electricity ($/kWh)": {"p10": 0.05, "p50": 0.10, "p90": 0.15},
        "Annual Interest Rate (%)": {"p10": 0.015, "p50": 0.025, "p90": 0.04}
      }
    }
    """
    selected_models: List[str]
    years: int = Field(..., gt=0, le=30)
    iterations: int = Field(..., gt=0, le=100_000)
    distributions: Dict[str, DistributionSettings]

# Pydantic Model for the forecast result
class SingleScenarioResult(BaseModel):
    scenario: str
    p10: List[float]
    p50: List[float]
    p90: List[float]

class ModelForecastResult(BaseModel):
    model_id: str
    dollars_per_km: List[SingleScenarioResult]
    dollars: List[SingleScenarioResult]
    npv: List[SingleScenarioResult]

class ForecastResponse(BaseModel):
    """
    The overall response from /forecast,
    including data for each selected model.
    """
    results: List[ModelForecastResult]


@app.post("/forecast", response_model=ForecastResponse)
def run_forecast(request: ForecastRequest):
    """
    Runs a Monte Carlo forecast for the given models with user-specified
    distribution settings (p10, p50, p90). Returns the resulting percentile
    lines (P10, P50, P90) for each scenario across the forecast horizon.
    """
    # 1) Validate that the requested models exist in data.models
    all_ids = set(data.models["ID"].unique())
    for mid in request.selected_models:
        if mid not in all_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Model ID '{mid}' not found in data.models"
            )

    # 2) Build a local dictionary that mimics how your app uses distribution_data
    #    Normally, your app might do create_dist -> metalog -> st.session_state,
    #    but let's do a minimal approach: we interpret p10/p50/p90 as a uniform range or something
    #    For a real integration, you'd replicate your distribution approach exactly.
    
    distribution_data = {}
    for param, dist_settings in request.distributions.items():
        # For simplicity, let's create a random array using a naive approach:
        # We'll interpret p10/p90 as min and max, ignoring p50 for now:
        # (In your real code, you'd do metalog or some other sampling approach.)
        low = dist_settings.p10
        high = dist_settings.p90

        # We'll create a random uniform distribution for (years+1)*iterations
        # Then reshape to [iterations, years+1].
        # This is just an example; you should replicate your real approach from 'analysis.py'.
        sample_size = (request.iterations, request.years + 1)
        samples = np.random.uniform(low=low, high=high, size=sample_size)
        distribution_data[param] = samples

    # 3) We'll replicate some logic from analysis.monte_carlo_forecast (simplified).
    #    We'll create Model objects in memory, run the simulation, and then compute some percentiles.
    
    # Build a local dictionary of Model objects for only the requested models
    simulation_models = {}
    for mid in request.selected_models:
        row = data.models.loc[data.models["ID"] == mid].squeeze()
        row_dict = row.to_dict()
        ID = row_dict.pop("ID")
        Fuel = row_dict.pop("Fuel")
        year_ = row_dict.pop("Year")
        Vehicle_Class = row_dict.pop("Veh Class")

        # Create the controls.Model object
        m = controls.Vehicle(
            ID=ID,
            Fuel=Fuel,
            Year=year_,
            Vehicle_Class=Vehicle_Class,
            theoretical_efficiency={
                "City": row_dict.get("City", None),
                "Combined": row_dict.get("Combined", None),
                "Highway": row_dict.get("Highway", None),
            }
        )
        simulation_models[ID] = m

    # 3B) We'll do a minimal replicate of the analysis. For example:
    # a) Annual Distance (km)
    km = distribution_data["Annual Distance (km)"]  # shape = [iterations, years+1]
    # b) Price of Gasoline ($/L)
    price_per_liter = distribution_data["Price of Gasoline ($/L)"]
    # c) Price of Electricity ($/kWh)
    #    We'll store negative so it matches your code's usage of "dollars_per_kWh = -price" if needed
    price_per_kWh = -distribution_data["Price of Electricity ($/kWh)"]
    # d) interest_rates (1 + r) for discounting
    interest_rates = 1.0 + distribution_data["Annual Interest Rate (%)"]

    # Convert interest rates to discount factors
    for i in range(interest_rates.shape[1]):
        interest_rates[:, i] **= i + 1

    # e) We'll run a minimal version of the run_simulation
    performance_modifier = 1.0

    for m_obj in simulation_models.values():
        # pick the correct price array based on fuel type
        if m_obj.Fuel == "Electricity":
            cost_array = price_per_kWh
        else:  # Gasoline
            # We might do something like converting from $/L -> $/kWh:
            # from your code: kWh_per_gal=33.41, gal_per_liter=0.2199692 => kWh_per_liter = kWh_per_gal*gal_per_liter
            # But let's keep it simple for demonstration:
            cost_array = -price_per_liter * analysis.liter_per_kWh

        m_obj.run_simulation(performance_modifier, cost_array, km, interest_rates)

    # 4) Summarize results in P10/P50/P90 lines for each scenario & measure over time
    #    We'll produce the same shape [years+1], so user can see the lines each year.
    
    final_results = []
    scenarios = ["City", "Combined", "Highway"]
    for ID, m_obj in simulation_models.items():
        model_record = ModelForecastResult(
            model_id=ID,
            dollars_per_km=[],
            dollars=[],
            npv=[]
        )

        for scenario in scenarios:
            # dollars_per_km is shape [iterations, years+1]
            dpk_data = m_obj.data["dollars_per_km"][scenario]
            if dpk_data is not None:
                p10_vals = np.percentile(dpk_data, 10, axis=0).tolist()
                p50_vals = np.percentile(dpk_data, 50, axis=0).tolist()
                p90_vals = np.percentile(dpk_data, 90, axis=0).tolist()
            else:
                # fallback if none
                p10_vals, p50_vals, p90_vals = [], [], []

            model_record.dollars_per_km.append(SingleScenarioResult(
                scenario=scenario,
                p10=p10_vals,
                p50=p50_vals,
                p90=p90_vals
            ))

            # dollars
            d_data = m_obj.data["dollars"][scenario]
            if d_data is not None:
                p10_vals = np.percentile(d_data, 10, axis=0).tolist()
                p50_vals = np.percentile(d_data, 50, axis=0).tolist()
                p90_vals = np.percentile(d_data, 90, axis=0).tolist()
            else:
                p10_vals, p50_vals, p90_vals = [], [], []
            
            model_record.dollars.append(SingleScenarioResult(
                scenario=scenario,
                p10=p10_vals,
                p50=p50_vals,
                p90=p90_vals
            ))

            # npv
            n_data = m_obj.data["npv"][scenario]
            if n_data is not None:
                p10_vals = np.percentile(n_data, 10, axis=0).tolist()
                p50_vals = np.percentile(n_data, 50, axis=0).tolist()
                p90_vals = np.percentile(n_data, 90, axis=0).tolist()
            else:
                p10_vals, p50_vals, p90_vals = [], [], []

            model_record.npv.append(SingleScenarioResult(
                scenario=scenario,
                p10=p10_vals,
                p50=p50_vals,
                p90=p90_vals
            ))

        final_results.append(model_record)

    return ForecastResponse(results=final_results)

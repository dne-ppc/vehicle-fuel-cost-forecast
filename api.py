from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from models import Simulation

app = FastAPI(title="Vehicle Operating Cost Forecast API")

# Create a global simulation instance.
# The Simulation __init__ reads data/models.csv and data/config.yaml.
simulation = Simulation()


class SimulationRunRequest(BaseModel):
    vehicle_ids: List[str]
    years: Optional[int] = 10
    iterations: Optional[int] = 10000


@app.get("/vehicles")
def list_vehicles():
    """
    Returns a list of all available models.
    """
    # simulation.models is a DataFrame loaded from CSV.
    models_df = simulation.models
    models_list = models_df.to_dict(orient="records")
    return models_list


@app.get("/distributions")
def list_distributions():
    """
    Returns the list of distribution configurations available.
    """
    dists = []
    for label, dist in simulation.distributions.items():
        dists.append(
            {
                "label": label,
                "min_value": dist.min_value,
                "max_value": dist.max_value,
                "p10": dist.p10,
                "p50": dist.p50,
                "p90": dist.p90,
                "step": dist.step,
                "boundedness": dist.boundedness,
            }
        )
    return dists


@app.post("/simulation/run")
def run_simulation(request: SimulationRunRequest):
    """
    Loads the selected model IDs into the simulation, runs the Monte Carlo forecast,
    and returns the forecast results for each model.
    """
    if not request.vehicle_ids:
        raise HTTPException(status_code=400, detail="No vehicle IDs provided.")

    # Update simulation parameters
    simulation.years = request.years
    simulation.iterations = request.iterations

    # Load vehicles for the selected model IDs and run the simulation forecast
    simulation.load_vehicles(request.vehicle_ids)
    simulation.monte_carlo_forecast()

    return simulation



@app.post("/simulation/sensitivity")
def simulation_sensitivity(request: SimulationRunRequest,percentile: Optional[int] = 50):
    """
    Runs an NPV sensitivity analysis for the current simulation using the given percentile (default: 50).
    Returns a list of sensitivity records.
    """
    if not request.vehicle_ids:
        raise HTTPException(status_code=400, detail="No vehicle IDs provided.")

    # Update simulation parameters
    simulation.years = request.years
    simulation.iterations = request.iterations

    # Load vehicles for the selected model IDs and run the simulation forecast
    simulation.load_vehicles(request.vehicle_ids)
    simulation.monte_carlo_forecast()

    df = simulation.npv_sensitivity(percentile)
    # Convert the DataFrame to a list of dictionaries
    sensitivity_results = df.to_dict(orient="records")
    return sensitivity_results

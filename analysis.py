from typing import Dict
import streamlit as st
import numpy as np
import pandas as pd
from metalog import metalog
import controls
import data
import itertools

kWh_per_gal = 33.41  # https://en.wikipedia.org/wiki/Gasoline_gallon_equivalent
gal_per_liter = 0.2199692
kWh_per_liter = kWh_per_gal * gal_per_liter
liter_per_kWh = 1 / kWh_per_liter


def create_data(
    dist,
    label,
    iterations,
    years,
    sensitivity_param=None,
    sensitivity_value=None,
    **kwargs,
):
    """
    Draws samples from a Metalog distribution for each year and iteration.

    If sensitivity_param matches the current label, instead of sampling from
    the Metalog distribution, this function uses a constant value (sensitivity_value)
    to simulate the parameter being fixed at its lower or upper bound.

    Args:
        dist (dict): The fitted Metalog distribution from st.session_state.
        label (str): The name of the distribution (e.g., "Price of Gasoline ($/L)").
        iterations (int): Number of Monte Carlo iterations.
        years (int): Number of years to forecast.
        sensitivity_param (str): Parameter name to override if it matches `label`.
        sensitivity_value (float): The forced value if sensitivity_param == label.
        **kwargs: Other arguments (unused here).

    Returns:
        np.ndarray: A 2D array of shape [iterations, years+1] with sampled or constant values.
    """
    end_year = years + 1
    if sensitivity_param == label:
        # If we are applying sensitivity, override with a constant
        values = np.full([iterations, end_year], sensitivity_value)
    else:
        values = metalog.r(m=dist, n=iterations * end_year).reshape(
            iterations, end_year
        )

    return values


def monte_carlo_forecast(years=10, iterations=1000, distributions=None, **kwargs):
    """
    Computes a Monte Carlo forecast for all selected models by sampling from
    multiple distributions (stored in st.session_state). The function:
      1. Samples from each distribution (e.g. fuel prices, interest rate).
      2. Applies these samples in `controls.Model.run_simulation` to compute cost arrays.
      3. Creates "comparison" model objects for each pair of selected models.

    The results (cost arrays) are stored back in st.session_state["simulation_models"].

    Args:
        years (int): Number of years to forecast.
        iterations (int): Number of Monte Carlo iterations per year.
        distributions (dict): Dictionary of parameter distributions to sample from.
        **kwargs: May include sensitivity overrides (sensitivity_param, sensitivity_value).
    """
    models: Dict[str, controls.Model] = st.session_state["simulation_models"]

    # Sample each distribution or set constant sensitivity value
    for label, dist in distributions.items():
        data_arr = create_data(dist, label, iterations, years, **kwargs)
        st.session_state.distribution_data[label] = data_arr

    # Optional performance_modifier usage, but for now it's 1.0
    performance_modifier = 1.0

    km = st.session_state.distribution_data["Annual Distance (km)"]
    price_per_liter = st.session_state.distribution_data["Price of Gasoline ($/L)"]
    dollars_per_kWh = -st.session_state.distribution_data[
        "Price of Electricity ($/kWh)"
    ]
    interest_rates = 1 + st.session_state.distribution_data["Annual Interest Rate (%)"]

    # Convert interest rates to discount factors over time
    for i in range(interest_rates.shape[1]):
        interest_rates[:, i] **= i + 1

    # Update each Model with new simulation data
    for id, model in models.items():
        if model.comparison:
            # Skip if it's already a comparison model
            continue

        if model.Fuel == "Electricity":
            model_dollars_per_kWh = dollars_per_kWh
        else:  # 'Gasoline'
            model_dollars_per_kWh = -price_per_liter * liter_per_kWh

        model.run_simulation(
            performance_modifier, model_dollars_per_kWh, km, interest_rates
        )

    # Create difference (comparison) models for each pair of selected models
    selections = st.session_state.selected_models
    combinations = list(itertools.combinations(selections, 2))
    for model_a_id, model_b_id in combinations:
        model = controls.Model.make_comparison(model_a_id, model_b_id, interest_rates)
        models[model.ID] = model

    st.session_state["simulation_models"] = models



def collect_npvs(sensitivities, row, treatment, percentile):
    """
    Collects the NPV data from each model and scenario at a given percentile.

    Args:
        sensitivities (list): A running list that accumulates dict entries for each scenario.
        row (pd.Series): Row in the dataframe describing the parameter and its bounds.
        treatment (str): One of 'Baseline', 'Low', or 'High' to indicate how the parameter was varied.
        percentile (int): The percentile (e.g., 50 for median) to compute across all simulations.
    """
    # print(row)
    for model_name, model in st.session_state["simulation_models"].items():
        for scenario, data in model.data["npv"].items():
            npv = np.percentile(data, percentile, axis=0).sum()

            sensitivity = {
                "Model": model_name,
                "Parameter": row["Parameter"],
                "Scenario": scenario,
                "Treatment": treatment,
                treatment: npv,
                "Lower Bound": None,
                "Upper Bound": None,
            }
            if treatment == "Low":
                sensitivity["Lower Bound"] = row["Lower Bound"]
            elif treatment == "High":
                sensitivity["Upper Bound"] = row["Upper Bound"]

            sensitivities.append(sensitivity)


def get_bounds(uid="distributions"):
    """
    Reads the user's current P10/P90 inputs from Streamlit state for each distribution
    and returns a DataFrame with columns:
        ['Parameter', 'Lower Bound', 'Upper Bound'].
    The 'Parameter' is the label you passed to create_dist(...).
    The 'Lower Bound' is the P10 value, and 'Upper Bound' is the P90 value.
    """
    bounds_list = []

    # Make sure 'uid' exists in session_state
    if uid in st.session_state:
        # Each 'label' is the parameter name used in create_dist(... label="some label" ...)
        for label in st.session_state[uid].keys():
            # Build the keys that create_triplet uses for P10/P90
            low_key = f"{uid}_{label}_low"   # e.g. "distributions_Price of Gasoline ($/L)_low"
            high_key = f"{uid}_{label}_high"

            # Check if those keys exist in session_state
            if low_key in st.session_state and high_key in st.session_state:
                # Grab the current user inputs for P10 / P90
                p10_val = st.session_state[low_key]
                p90_val = st.session_state[high_key]

                # Add a row with the parameter name and its P10/P90
                bounds_list.append({
                    "Parameter": label, 
                    "Lower Bound": p10_val, 
                    "Upper Bound": p90_val
                })

    return pd.DataFrame(bounds_list)

def npv_sensitivity(percentile=50):
    """
    Computes the NPV sensitivity by:
      1. Getting the parameter bounds (data.get_bounds()).
      2. Setting each parameter to its Low or High bound.
      3. Running the forecast (monte_carlo_forecast).
      4. Collecting NPVs across these runs at a specified percentile.

    Restores the baseline scenario after each parameter is tested.

    Args:
        percentile (int): The percentile to compute (e.g., 50 for median).

    Returns:
        pd.DataFrame: A pivoted DataFrame showing how NPV changes under Low/High
                      parameter values for each model and scenario.
    """
    bounds_df = get_bounds()
    sensitivities = []

    # Baseline run
    for _, row in bounds_df.iterrows():
        collect_npvs(sensitivities, row, "Baseline", percentile)

    # Low/High runs for each parameter
    for _, row in bounds_df.iterrows():
        monte_carlo_forecast(
            **st.session_state,
            sensitivity_param=row["Parameter"],
            sensitivity_value=row["Lower Bound"],
        )
        collect_npvs(sensitivities, row, "Low", percentile)

        monte_carlo_forecast(
            **st.session_state,
            sensitivity_param=row["Parameter"],
            sensitivity_value=row["Upper Bound"],
        )
        collect_npvs(sensitivities, row, "High", percentile)

    # Restore baseline
    monte_carlo_forecast(**st.session_state)

    df = pd.DataFrame(sensitivities)

    baseline = (
        df.loc[df.Treatment == "Baseline"]
        .dropna(axis=1, how="all")
        .drop(columns=["Treatment"])
    )
    low = (
        df.loc[df.Treatment == "Low"]
        .dropna(axis=1, how="all")
        .drop(columns=["Treatment"])
    )
    high = (
        df.loc[df.Treatment == "High"]
        .dropna(axis=1, how="all")
        .drop(columns=["Treatment"])
    )

    df = baseline.merge(low, on=["Model", "Parameter", "Scenario"]).merge(
        high, on=["Model", "Parameter", "Scenario"]
    )

    # for purpose of plotting we need to switch so that bars have correct
    # relationship to baseline
    mask = df.Low > df.High
    df.loc[mask, ["High", "Low"]] = df.loc[mask, ["Low", "High"]].values
    df.loc[mask, ["Lower Bound", "Upper Bound"]] = df.loc[
        mask, ["Upper Bound", "Lower Bound"]
    ].values
    df["Range"] = df["High"] - df["Low"]
    df.sort_values(["Model", "Scenario", "Range"], ascending=True, inplace=True)

    return df

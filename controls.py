from typing import Dict
from dataclasses import dataclass

import streamlit as st
import numpy as np
from metalog import metalog

from data import models
import plotting

np.float_ = np.float64


@dataclass
class Model:
    """
    Represents a vehicle model, including fuel type, year, class,
    and data arrays for different cost measures (dollars_per_km, dollars, npv).

    Attributes:
        ID (str): Identifier for the model (or model comparison).
        Fuel (str): Type of fuel (e.g., "Electricity", "Gasoline").
        Year (int): Model year.
        Vehicle_Class (str): Classification of the vehicle (e.g., "Sedan", "Truck").
        theoretical_efficiency (Dict[str, float]): Efficiency for City, Combined, Highway.
        total_cost_data (Dict[str, np.array]): Possibly unused here, placeholder for cost data.
        data (Dict[str, Dict[str, np.array]]): Contains distribution arrays for the model's costs.
        comparison (bool): Flag for whether this instance represents a comparison between two models.
    """

    ID: str
    Fuel: str = None
    Year: int = None
    Vehicle_Class: str = None
    theoretical_efficiency: Dict[str, float] = None
    total_cost_data: Dict[str, np.array] = None
    data: Dict[str, Dict[str, np.array]] = None
    comparison: bool = False

    def __post_init__(self):
        """
        Initializes the data structure for storing cost arrays under each scenario
        (City, Combined, Highway). If theoretical_efficiency is not set, a default
        dict with None values is used. This method is automatically called
        after object creation.
        """
        scenarios = {"City": None, "Combined": None, "Highway": None}
        self.data = {
            "dollars_per_km": scenarios.copy(),
            "dollars": scenarios.copy(),
            "npv": scenarios.copy(),
        }
        if self.theoretical_efficiency is None:
            self.theoretical_efficiency = {
                "City": None,
                "Combined": None,
                "Highway": None,
            }

    @classmethod
    def make_comparison(cls, model_a_id, model_b_id, interest_rates):
        """
        Creates a special Model instance that represents the difference in cost
        between two existing models (model_a and model_b).

        The new Model's cost data is computed as the cost for model_a minus
        the cost for model_b under each scenario, for each year.

        Args:
            model_a_id (str): Identifier for the first model.
            model_b_id (str): Identifier for the second model.
            interest_rates (np.array): Precomputed interest rates per year.

        Returns:
            Model: A new Model instance representing (model_a - model_b) cost comparison.
        """
        models: Dict[str, Model] = st.session_state["simulation_models"]
        model_a = models[model_a_id]
        model_b = models[model_b_id]

        ID = f"{model_a_id} vs {model_b_id}"
        Fuel = f"{model_a.Fuel} vs {model_b.Fuel}"

        model = cls(ID=ID, Fuel=Fuel, comparison=True)

        for scenario in model.theoretical_efficiency.keys():
            a_dollars_per_km = model_a.data["dollars_per_km"][scenario]
            b_dollars_per_km = model_b.data["dollars_per_km"][scenario]

            a_dollars = model_a.data["dollars"][scenario]
            b_dollars = model_b.data["dollars"][scenario]

            dollar_difference = a_dollars - b_dollars

            model.data["dollars_per_km"][scenario] = a_dollars_per_km - b_dollars_per_km
            model.data["dollars"][scenario] = dollar_difference
            model.data["npv"][scenario] = dollar_difference / interest_rates

        return model

    def run_simulation(self, performance_modifier, dollars_per_kWh, km, interest_rates):
        """
        Performs a single "simulation run" for each scenario. Sets
        the cost data arrays (dollars_per_km, dollars, npv) based on:
          - Performance modifier
          - Electricity or gasoline price in dollars_per_kWh
          - Annual distance traveled (km)
          - Interest rates array for each year

        Args:
            performance_modifier (float or np.array): Factor to modify the theoretical efficiency.
            dollars_per_kWh (float or np.array): Energy cost, either electric (kWh) or gasoline (converted to kWh basis).
            km (np.array): Annual distance traveled per iteration and year.
            interest_rates (np.array): (1 + annual interest rate) raised to the year index for discounting.
        """
        for scenario, kWh_per_km in self.theoretical_efficiency.items():
            # Multiply by performance modifier if needed. Currently set to 1.0 by default in analysis.py
            dollars_per_km = kWh_per_km * dollars_per_kWh
            dollars = dollars_per_km * km
            self.data["dollars_per_km"][scenario] = dollars_per_km
            self.data["dollars"][scenario] = dollars
            self.data["npv"][scenario] = dollars / interest_rates


def select_models():
    """
    Renders a UI widget to allow multiple model selections.
    Generates Model instances for each selected model and stores them
    in session state as 'simulation_models'.

    Returns:
        list: A list of selected model IDs.
    """
    selected_models = st.multiselect(
        label="Select the models", options=models.ID, max_selections=2
    )

    if not select_models:
        st.session_state.simulation_models = {}
        st.session_state.selected_models = []

    processed_models = {}
    for selected in selected_models:
        model = models.loc[models.ID == selected].squeeze()
        model_dict = model.to_dict()
        ID = model_dict.pop("ID")
        Fuel = model_dict.pop("Fuel")
        year = model_dict.pop("Year")
        Vehicle_Class = model_dict.pop("Veh Class")

        processed_models[ID] = Model(
            ID=ID,
            Fuel=Fuel,
            Year=year,
            Vehicle_Class=Vehicle_Class,
            theoretical_efficiency=model_dict,
        )

    st.session_state.simulation_models = processed_models
    st.session_state.selected_models = selected_models

    return selected_models


def create_triplet(
     uid, label, min_value, max_value, low, medium, high, step=0.01, labels=None, **kwargs
):
    if labels is None:
        labels = ['P10','P50','P90']
    
    left, middle, right = st.columns(3)
    low_key = f"{uid}_{label}_low"
    medium_key = f"{uid}_{label}_medium"
    high_key = f"{uid}_{label}_high"

    margin = max(medium * 0.1, step * 3)

    def validate_low():
        low = st.session_state[low_key]
        medium = st.session_state[medium_key]
        if low + margin >= medium:
            st.session_state[low_key] = medium - margin

    def validate_medium():
        low = st.session_state[low_key]
        medium = st.session_state[medium_key]
        high = st.session_state[high_key]

        if medium - margin <= low:
            st.session_state[medium_key] = low + margin

        if medium + margin >= high:
            st.session_state[medium_key] = high - margin

    def validate_high():
        high = st.session_state[high_key]
        medium = st.session_state[medium_key]
        if high - margin <= medium:
            st.session_state[high_key] = high + margin

    with left:
        low = st.number_input(
            labels[0],
            min_value=min_value,
            max_value=max_value,
            value=low,
            step=step,
            key=low_key,
            on_change=validate_low,
        )
    with middle:
        medium = st.number_input(
            labels[1],
            min_value=min_value,
            max_value=max_value,
            value=medium,
            step=step,
            key=medium_key,
            on_change=validate_medium,
        )
    with right:
        high = st.number_input(
            labels[2],
            min_value=min_value,
            max_value=max_value,
            value=high,
            step=step,
            key=high_key,
            on_change=validate_high,
        )
    return low, medium, high


def create_dist(
    uid, label, min_value, max_value, p10, p50, p90, step=0.01, probs=None, **kwargs
):
    """
    Creates and fits a Metalog distribution based on user-defined P10, P50, and P90 values.
    Renders input widgets (number_input) for adjusting these percentile values.

    Args:
        uid (str): An identifier used to store and reference the distribution in session_state.
        label (str): The display label for the distribution (e.g., 'Price of Gasoline').
        min_value (float): The minimum possible value of the distribution.
        max_value (float): The maximum possible value of the distribution.
        p10 (float): The initial guess for 10th percentile.
        p50 (float): The initial guess for 50th percentile.
        p90 (float): The initial guess for 90th percentile.
        step (float): Increment for the Streamlit number input.
        probs (list): The probabilities corresponding to p10, p50, p90 for Metalog fitting.
        **kwargs: Additional arguments that might be passed in but are unused here.
    """
    if probs is None:
        probs = [0.1, 0.5, 0.9]

    if uid not in st.session_state:
        st.session_state[uid] = {}

    if label not in st.session_state:
        st.session_state[uid][label] = {}

    print(f"{uid=}, {label=}, {min_value=}, {max_value=}, {p10=}, {p50=}, {p90=}")

    st.subheader(label)
    p10,p50,p90 = create_triplet( uid, label, min_value, max_value, p10, p50, p90, step)

    try:
        dist = metalog.fit(
            x=[p10, p50, p90],
            boundedness="b",
            bounds=[min_value - step, max_value + step],
            term_limit=3,
            probs=probs,
        )
        st.plotly_chart(
            plotting.create_dist_plot(dist),
            use_container_width=True,
            key=f"{uid}_{label}",
        )
    except Exception as e:
        st.error(f"Error fitting installation cost distribution: {e}")

    st.session_state[uid][label] = dist
    st.divider()

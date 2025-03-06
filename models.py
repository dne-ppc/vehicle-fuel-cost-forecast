# models.py
from typing import Dict, List, Optional, TypeVar
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pydantic import BaseModel, Field
import pydantic_numpy.typing as pnd
from metalog import metalog
import yaml
import itertools


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
kWh_per_gal = 33.41  # https://en.wikipedia.org/wiki/Gasoline_gallon_equivalent
gal_per_liter = 0.2199692
kWh_per_liter = kWh_per_gal * gal_per_liter
liter_per_kWh = 1 / kWh_per_liter


class ScenarioResult(BaseModel):
    """
    A small Pydantic model holding P10/P50/P90 lines for a given scenario.
    """

    scenario: str
    p10: List[float]
    p50: List[float]
    p90: List[float]


class Vehicle(BaseModel):
    """
    A unified "VehicleModel" that holds:
      - Basic vehicle info (ID, Fuel, etc.)
      - Theoretical efficiency per scenario
      - A private dictionary for simulation results
      - Methods to run the simulation and produce P10/P50/P90 outputs
    """

    # Basic fields
    ID: str
    Fuel: Optional[str] = None
    Year: Optional[int] = None
    Vehicle_Class: Optional[str] = Field(None, alias="Veh Class")
    comparison: bool = False

    # Theoretical efficiency in kWh/km for City, Combined, Highway
    # (Used by run_simulation to compute actual cost arrays.)
    theoretical_efficiency: Dict[str, Optional[float]] = Field(default_factory=dict)

    # Because Pydantic tries to validate everything, we store NumPy arrays in a private attribute.
    # By default, these won't appear in JSON output.
    class Config:
        underscore_attrs_are_private = True
        allow_population_by_field_name = (
            True  # So "Veh Class" can map to "Vehicle_Class" if needed
        )

    # This dictionary maps "dollars_per_km", "dollars", "npv" → scenario → array
    # We'll init it in __init__ so each Model instance has fresh structures.
    data: Dict[str, Dict[str, Optional[pnd.NpNDArray]]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize empty arrays for each measure-scenario
        measures = ["dollars_per_km", "dollars", "npv"]
        scenarios = ["City", "Combined", "Highway"]
        self.data = {measure: {scn: None for scn in scenarios} for measure in measures}

    @classmethod
    def make_comparison(
        cls, model_a: "Vehicle", model_b: "Vehicle", interest_rates: np.array
    ) -> "Vehicle":
        """
        Creates a new VehicleModel representing the difference (model_a - model_b)
        in all cost measures for each scenario. This is the main way to compare
        two existing models after their costs have been computed.

        The new VehicleModel will have:
          - ID = "A vs B"
          - Fuel = "A_fuel vs B_fuel"
          - comparison = True
          - arrays:
            dollars_per_km[scenario] = (model_a - model_b)
            dollars[scenario] = (model_a - model_b)
            npv[scenario] = (model_a - model_b)
        """
        ID = f"{model_a.ID} vs {model_b.ID}"
        Fuel = f"{model_a.Fuel} vs {model_b.Fuel}"
        new_model = cls(
            ID=ID,
            Fuel=Fuel,
            comparison=True,
            # This comparison won't rely on efficiency, so we can leave these blank
            theoretical_efficiency={"City": None, "Combined": None, "Highway": None},
        )

        for scenario in new_model.theoretical_efficiency.keys():
            a_dollars_per_km = model_a.data["dollars_per_km"][scenario]
            b_dollars_per_km = model_b.data["dollars_per_km"][scenario]

            a_dollars = model_a.data["dollars"][scenario]
            b_dollars = model_b.data["dollars"][scenario]

            dollar_difference = a_dollars - b_dollars

            new_model.data["dollars_per_km"][scenario] = (
                a_dollars_per_km - b_dollars_per_km
            )
            new_model.data["dollars"][scenario] = dollar_difference
            new_model.data["npv"][scenario] = dollar_difference / interest_rates

        return new_model

    def run_simulation(
        self,
        performance_modifier: float,
        dollars_per_kWh: np.ndarray,  # shape [iterations, years+1]
        km: np.ndarray,  # shape [iterations, years+1]
        interest_rates: np.ndarray,  # shape [iterations, years+1]
    ):
        """
        Performs a single "simulation run" for each scenario.
        Stores 2D arrays (iterations × years+1) in _data_arrays for:
            - 'dollars_per_km'
            - 'dollars'
            - 'npv'
        matching your original logic from controls.Model.
        """
        # Example adaptation from your existing code:
        # theoretical_efficiency: { 'City': 0.16, 'Combined': 0.18, 'Highway': 0.20 }, etc.
        for scenario, kWh_per_km in self.theoretical_efficiency.items():
            if kWh_per_km is None:
                continue

            # Multiply by performance if needed
            # Normally performance_modifier=1.0, but we keep it flexible
            eff = kWh_per_km * performance_modifier

            # cost per km
            dollars_per_km_array = eff * dollars_per_kWh  # shape [iter, yrs+1]
            self.data["dollars_per_km"][scenario] = dollars_per_km_array

            # total cost
            dollars_array = dollars_per_km_array * km
            self.data["dollars"][scenario] = dollars_array

            # discounting
            npv_array = dollars_array / interest_rates
            self.data["npv"][scenario] = npv_array

    def to_forecast_result(
        self,
        low_percentile: float = 10,
        median_percentile: float = 50,
        high_percentile: float = 90,
    ) -> Dict[str, List[ScenarioResult]]:
        """
        Converts the raw arrays in _data_arrays to a structure of
        measure -> list of ScenarioResult(P10/P50/P90).
        This effectively merges the old 'ModelForecastResult' concept
        into the same class, letting you produce the final JSON-friendly output.
        """
        from numpy import percentile

        scenarios = ["City", "Combined", "Highway"]
        measures = ["dollars_per_km", "dollars", "npv"]

        results = {}
        for measure in measures:
            scenario_results = []
            for scn in scenarios:
                arr = self.data[measure][scn]
                if arr is None:
                    # no data for that scenario
                    scenario_results.append(
                        ScenarioResult(scenario=scn, p10=[], p50=[], p90=[])
                    )
                else:
                    p10 = percentile(arr, low_percentile, axis=0).tolist()
                    p50 = percentile(arr, median_percentile, axis=0).tolist()
                    p90 = percentile(arr, high_percentile, axis=0).tolist()
                    scenario_results.append(
                        ScenarioResult(scenario=scn, p10=p10, p50=p50, p90=p90)
                    )
            results[measure] = scenario_results
        return results


class Distribution(BaseModel):
    """
    Stores the basic configuration for a single parameter's Metalog distribution.
    For example: Price of Gasoline ($/L), with min_value, max_value, p10, p50, p90, etc.
    """

    label: str
    min_value: float
    max_value: float
    p10: float
    p50: float
    p90: float
    step: float = 0.01
    boundedness: str = "b"  # 'b' for two-sided bounding in metalog

    def create_data(
        self,
        iterations: int,
        years: int,
        sensitivity_param: Optional[str] = None,
        sensitivity_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draws samples from the distribution for `label` using Metalog.
        If `sensitivity_param == label`, we override the distribution with a constant
        (`sensitivity_value`) for all samples. Returns a shape [iterations, years+1].
        """

        # If we're doing a "sensitivity" run, just fill everything with a constant
        if sensitivity_param == self.label and sensitivity_value is not None:
            return np.full((iterations, years + 1), sensitivity_value, dtype=float)

        # Otherwise, sample from the metalog distribution
        if not metalog:
            raise RuntimeError("metalog library is not installed or not imported")

        # Fit the metalog from the current (p10, p50, p90) config
        dist = metalog.fit(
            x=[self.p10, self.p50, self.p90],
            boundedness=self.boundedness,
            bounds=[self.min_value - self.step, self.max_value + self.step],
            term_limit=3,
            probs=[0.1, 0.5, 0.9],
        )
        # Sample (iterations*(years+1)) points
        values = metalog.r(dist, n=iterations * (years + 1))
        return values.reshape((iterations, years + 1))

    def create_controls(self) -> None:
        """
        A Streamlit-based method that replicates the logic of the old 'controls.create_dist'.
        It displays number inputs for the distribution's p10/p50/p90, updates them in session,
        fits the metalog, and plots the PDF/CDF in real-time.
        """

        st.subheader(self.label)
        p10_key = f"{self.label}_low"
        p50_key = f"{self.label}_medium"
        p90_key = f"{self.label}_high"

        col1, col2, col3 = st.columns(3)
        with col1:
            low_val = st.number_input("P10", value=self.p10, key=p10_key)
        with col2:
            med_val = st.number_input("P50", value=self.p50, key=p50_key)
        with col3:
            high_val = st.number_input("P90", value=self.p90, key=p90_key)

        self.p10 = low_val
        self.p50 = med_val
        self.p90 = high_val

        try:
            dist = metalog.fit(
                x=[low_val, med_val, high_val],
                boundedness=self.boundedness,
                bounds=[self.min_value - self.step, self.max_value + self.step],
                term_limit=3,
                probs=[0.1, 0.5, 0.9],
            )
            fig = self.create_dist_plot(dist)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fitting distribution for '{self.label}': {e}")

        st.divider()

    def create_dist_plot(self, m: str):
        """
        Creates a distribution plot (PDF + CDF) from a Metalog object.

        Args:
            m (str): A Metalog distribution dictionary returned by metalog.fit().

        Returns:
            go.Figure: A Plotly figure with two traces: PDF (left Y-axis) and CDF (right Y-axis).
        """
        quantiles = m["M"].iloc[:, 1]
        pdf_values = m["M"].iloc[:, 0]
        cdf_values = m["M"]["y"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=pdf_values / sum(pdf_values),
                mode="lines",
                name="PDF",
                line=dict(color="blue"),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles,
                y=cdf_values,
                mode="lines",
                name="CDF",
                line=dict(color="red", dash="dash"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Value"),
            yaxis=dict(title="PDF", title_font_color="blue", tickfont_color="blue"),
            yaxis2=dict(
                title="CDF",
                title_font_color="red",
                tickfont_color="red",
                overlaying="y",
                side="right",
            ),
            legend=dict(x=0, y=1.1, orientation="h"),
            template="plotly",
            hovermode="x unified",
        )
        return fig


class Simulation(BaseModel):
    """
    This class orchestrates the creation of Model A, Model B, and their comparison.
    It takes two model IDs, looks up the data rows, constructs the Model objects,
    and (optionally) runs the simulation logic.
    """

    years: int = 10
    iterations: int = 10000

    vehicles: Optional[Dict[str, Vehicle]] = Field(default_factory=dict)
    distributions: Optional[Dict[str, Distribution]] = Field(default_factory=dict)
    distribution_data: Optional[Dict[str, pnd.NpNDArray]] = Field(default_factory=dict)
    models: PandasDataFrame = None

    def __init__(self, **data):
        super().__init__(**data)
        self.models = pd.read_csv("data/models.csv")
        for distribution_ in self.get_config()["distributions"]:
            distribution = Distribution(**distribution_)

            self.distributions[distribution.label] = distribution

    def get_config(self):
        with open("data/config.yaml", "r") as f:
            return yaml.safe_load(f)

    def load_vehicles(self, ids):

        self.vehicles = {}

        for id_ in ids:
            row = self.models.loc[self.models.ID == id_].squeeze()
            row_dict = row.to_dict()
            ID = row_dict.pop("ID")
            fuel = row_dict.pop("Fuel", None)
            year_ = row_dict.pop("Year", None)
            vehicle_class = row_dict.pop("Veh Class", None)
            # We'll assume the rest are city/combined/highway columns or other attributes
            # that map to theoretical_efficiency:
            theory_eff = {
                k: row_dict[k] for k in ["City", "Combined", "Highway"] if k in row_dict
            }
            self.vehicles[id_] = Vehicle(
                ID=ID,
                Fuel=fuel,
                Year=year_,
                Vehicle_Class=vehicle_class,
                theoretical_efficiency=theory_eff,
            )

    def create_controls(self):
        col1, col2 = st.columns(2)

        with col1:
            self.years = st.number_input(
                "Years", min_value=1, max_value=20, value=10, step=1
            )

        with col2:
            self.iterations = st.number_input(
                "Iterations", min_value=1, max_value=10000, value=10000, step=100
            )

        col1, col2 = st.columns(2)

        with col1:
            selected_model_a = st.selectbox(
                label="Select Model A",
                options=self.models.ID.tolist(),  # Ensure it's a list of model IDs
                key="model_a",
            )

        with col2:
            selected_model_b = st.selectbox(
                label="Select Model B", options=self.models.ID.tolist(), key="model_b"
            )

        selected_models = []
        if selected_model_a:
            selected_models.append(selected_model_a)
        if selected_model_b:
            selected_models.append(selected_model_b)

        self.load_vehicles(selected_models)

        st.divider()
        for name, dist in self.distributions.items():
            dist.create_controls()

    def monte_carlo_forecast(self, **kwargs):
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

        if not self.vehicles:
            return

        for label, dist in self.distributions.items():
            self.distribution_data[label] = dist.create_data(
                self.iterations, self.years, **kwargs
            )

        # Optional performance_modifier usage, but for now it's 1.0
        performance_modifier = 1.0

        km = self.distribution_data["Annual Distance (km)"]
        price_per_liter = self.distribution_data["Price of Gasoline ($/L)"]
        dollars_per_kWh = self.distribution_data["Price of Electricity ($/kWh)"]
        interest_rates = 1 + self.distribution_data["Annual Interest Rate (%)"]

        # Convert interest rates to discount factors over time
        for i in range(interest_rates.shape[1]):
            interest_rates[:, i] **= i + 1

        # Update each Model with new simulation data
        for id, vehicle in self.vehicles.items():
            if vehicle.comparison:
                # Skip if it's already a comparison model
                continue

            if vehicle.Fuel == "Electricity":
                model_dollars_per_kWh = -dollars_per_kWh
            else:  # 'Gasoline'
                model_dollars_per_kWh = -price_per_liter * liter_per_kWh

            vehicle.run_simulation(
                performance_modifier, model_dollars_per_kWh, km, interest_rates
            )

        # Create difference (comparison) models for each pair of selected models
        vehicles = list(self.vehicles.values())[0:2]
        combinations = list(itertools.combinations(vehicles, 2))
        for model_a, model_b in combinations:
            vehicle = Vehicle.make_comparison(model_a, model_b, interest_rates)
            self.vehicles[vehicle.ID] = vehicle

    def get_bounds(self):
        """
        Reads the user's current P10/P90 inputs from Streamlit state for each distribution
        and returns a DataFrame with columns:
            ['Parameter', 'Lower Bound', 'Upper Bound'].
        The 'Parameter' is the label you passed to create_dist(...).
        The 'Lower Bound' is the P10 value, and 'Upper Bound' is the P90 value.
        """
        bounds_list = []

        for param, dist in self.distributions.items():
            bounds_list.append(
                {
                    "Parameter": dist.label,
                    "Lower Bound": dist.p10,
                    "Upper Bound": dist.p90,
                }
            )

        return pd.DataFrame(bounds_list)
    
    def collect_npvs(self,sensitivities, row, treatment, percentile):
        """
        Collects the NPV data from each model and scenario at a given percentile.

        Args:
            sensitivities (list): A running list that accumulates dict entries for each scenario.
            row (pd.Series): Row in the dataframe describing the parameter and its bounds.
            treatment (str): One of 'Baseline', 'Low', or 'High' to indicate how the parameter was varied.
            percentile (int): The percentile (e.g., 50 for median) to compute across all simulations.
        """
        # print(row)
        for vehicle_name, vehicle in self.vehicles.items():
            for scenario, data in vehicle.data["npv"].items():
                npv = np.percentile(data, percentile, axis=0).sum()

                sensitivity = {
                    "Model": vehicle_name,
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


    def npv_sensitivity(self, percentile=50):
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
        bounds_df = self.get_bounds()
        sensitivities = []

        # Baseline run
        for _, row in bounds_df.iterrows():
            self.collect_npvs(sensitivities, row, "Baseline", percentile)

        # Low/High runs for each parameter
        for _, row in bounds_df.iterrows():
            self.monte_carlo_forecast(
                sensitivity_param=row["Parameter"],
                sensitivity_value=row["Lower Bound"],
            )
            self.collect_npvs(sensitivities, row, "Low", percentile)

            self.monte_carlo_forecast(
                sensitivity_param=row["Parameter"],
                sensitivity_value=row["Upper Bound"],
            )
            self.collect_npvs(sensitivities, row, "High", percentile)

        # Restore baseline
        self.monte_carlo_forecast()

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

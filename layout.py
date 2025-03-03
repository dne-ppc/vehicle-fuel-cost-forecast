import streamlit as st
import data
import controls
import analysis
import plotting
import pandas as pd
import plotly.express as px


class Layout:
    """
    Encapsulates different layout sections and view functions for the Streamlit application.
    Each static method corresponds to a specific tab or section of the UI.
    """

    @staticmethod
    def models(*args, **kwargs):
        # """
        # Displays a DataFrame containing vehicle models.

        # Currently commented-out code shows how this method could be expanded
        # to visualize model performance via Plotly charts (e.g. bar charts showing
        # efficiency metrics). This method can be used if you want to show a list
        # of available models in the UI.
        # """
        # st.dataframe(data.models, use_container_width=True, height=900, hide_index=True)

        # """
        # Allows the user to filter the 'data.models' DataFrame by vehicle class, year,
        # and then select either the best (lowest kWh/km) or worst (highest kWh/km) models.

        # A single Plotly bar chart is created to visualize these models, grouped by fuel type.
        # """
        st.header("Models: Find Best/Worst-Performing Vehicles")
        st.markdown(
            """
            **Usage**: This tab helps you explore and filter the dataset of available vehicles, 
            so you can identify the best (lowest) or worst (highest) kWh/km in a given driving scenario. 
            Choose a vehicle class and year, select how many top or bottom vehicles you want, 
            then view the resulting bar chart comparing their efficiencies.
            """
        )
        st.subheader("Find Best or Worst Efficiency Models")

        # Get unique classes and years from the data
        classes = sorted(list(data.models["Veh Class"].unique()))
        years = sorted(list(data.models["Year"].unique()))

        # UI Inputs

        cols = st.columns(5)

        with cols[0]:
            selected_class = st.selectbox("Select Vehicle Class", classes)
        with cols[1]:
            selected_year = st.selectbox("Select Year", years, index=None)
        with cols[2]:
            best_worst = st.selectbox("Show Best or Worst?", ["Best", "Worst"])
        with cols[3]:
            top_n = st.number_input(
                "Number of models to display:", min_value=1, max_value=20, value=5
            )
        with cols[4]:
            scenario = st.selectbox(
                "Select Scenario",
                ["Combined", "City", "Highway"],
                key="select_models_scenario",
            )

        mask = data.models["Veh Class"] == selected_class

        if selected_year:
            mask &= data.models["Year"] == selected_year
        # Filter the DataFrame by the user’s selection
        df_filtered = data.models[mask]

        # Group the filtered data by fuel type and retrieve top/bottom N
        subsets = []
        for fuel_type, group in df_filtered.groupby("Fuel"):
            if best_worst == "Best":
                # 'nsmallest' picks the lowest kWh/km
                subset = group.nsmallest(top_n, scenario)
            else:
                # 'nlargest' picks the highest kWh/km
                subset = group.nlargest(top_n, scenario)

            subsets.append(subset)

        # If we found any matching vehicles, plot them
        if subsets:
            df_plot = pd.concat(subsets)
            fig = px.bar(
                df_plot,
                x="ID",
                y=scenario,
                color="Fuel",
                barmode="group",
                title=f"{best_worst} {top_n} by kWh/km for {selected_class} in {selected_year}",
                hover_data=["Year", "Veh Class", "Fuel"],
            )
            fig.update_layout(height=700, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models found for your selection.")

    @staticmethod
    def controls(*args, **kwargs):
        """
        Renders UI components for selecting models and specifying parameters
        for the simulation (e.g. years, iterations, and distributions).

        Returns:
            selections (list): A list of selected model IDs.
        """
        st.header("Simulation Controls: Select Models & Distributions")
        st.markdown(
            """
            **Usage**: Use this sidebar-like tab to pick which models you want to analyze, 
            choose how many years and iterations to simulate, and set the parameter distributions 
            (price ranges, interest rates, etc.). These settings apply to the other tabs.
            """
        )
        st.divider()
        selections = controls.select_models()

        col1, col2 = st.columns(2)

        with col1:
            years = st.number_input(
                "Years", min_value=1, max_value=20, value=10, step=1
            )
            st.session_state.years = years

        with col2:
            iterations = st.number_input(
                "Iterations", min_value=1, max_value=10000, value=10000, step=100
            )
            st.session_state.iterations = iterations

        st.divider()
        for dist in data.get_config()["distributions"]:
            controls.create_dist("distributions", **dist)

        return selections

    @staticmethod
    def forecasts(selections, *args, **kwargs):
        """
        Displays forecast results for the selected models. Allows the user to choose
        a measure (dollars_per_km, dollars, npv) and whether it should be shown cumulatively.

        - Runs a Monte Carlo forecast (analysis.monte_carlo_forecast) if models are selected.
        - Plots forecast results (plotting.create_forecast_plot).
        - Offers a percentile selector to see specific forecast quantiles.

        Args:
            selections (list): The list of selected model IDs.
        """
        st.header("Forecasts: Explore Cost Projections Over Time")
        st.markdown(
            """
            **Usage**: Select one or two models in the Controls tab, 
            then choose which measure (dollars_per_km, dollars, or npv) to plot 
            as a time series. This shows median and percentile bands (based on your selection) 
            for City, Combined, and Highway scenarios, letting you see how 
            costs may evolve over the forecast horizon.
            """
        )

        if selections:

            analysis.monte_carlo_forecast(**st.session_state)

            plotting.create_forecast_plot()
        else:
            st.info("Select models to create forecasts")

    @staticmethod
    def npv_sensitivity(selections, *args, **kwargs):
        """
        Displays an NPV sensitivity analysis for the selected models.
        This calculates how changes in parameters (e.g., fuel price, interest rate)
        affect the NPV of fuel over time.

        - Uses analysis.npv_sensitivity to compute scenarios.
        - Plots a 'tornado chart' for each model-scenario combination.

        Args:
            selections (list): The list of selected model IDs.
        """
        st.header("NPV Sensitivity: Tornado Charts")
        st.markdown(
            """
            **Usage**: Pick one or two models in the Controls tab, then run a 
            sensitivity analysis on the Net Present Value (NPV). This tab recalculates 
            the NPV while fixing each parameter (fuel price, interest rate, etc.) 
            to its low or high bound, showing which parameters have the most 
            significant impact and in which direction (increasing or decreasing NPV).

            **Note**
            - Negative NPV is the cost of the fuel for that vehicle choice 
            - Comparison (A vs B) shows the NPV of A - B 
            - Positive values for a comparison show that choosing A over B has positive value to you.
            - The values beside the bar show the value the parameter was set to for the simulation
            """
        )
        if selections:

            col1, col2 = st.columns(2)
            with col1:
                scenario = st.selectbox(
                    "Pick the driving scenario", options=["Combined", "City", "Highway"]
                )
            with col2:
                percentile = st.number_input(
                    "Select Percentile for Baseline",
                    min_value=1,
                    max_value=100,
                    value=50,
                    step=1,
                    key="select_sensitivity_percentile",
                )

            sensitives = analysis.npv_sensitivity(percentile)

            mask = sensitives["Scenario"] == scenario

            grouper = sensitives.loc[mask].groupby("Model")
            tabs = st.tabs(grouper.groups)

            for i, (model, model_data) in enumerate(grouper):
                with tabs[i]:
                    plotting.create_tornado_figure(model_data, model)
        else:
            st.info("Select models to create sensitivity")

    @staticmethod
    def npv_distribution(selections, *args, **kwargs):
        """
        Displays a PDF & CDF histogram for total NPV values for each model,
        based on the selected scenario (City, Combined, or Highway).
        Each model is shown on its own tab.

        Args:
            selections (list): List of selected model IDs.
        """
        st.header("NPV Distribution: PDF/CDF Analysis")
        st.markdown(
            """
            **Usage**: After running the Monte Carlo simulation, 
            examine the probability distribution of total NPV (sum of all forecast years). 
            Each model’s PDF and CDF are plotted with vertical lines for the probabilities 
            you choose to highlight. This helps you understand the risk spread of outcomes.
            """
        )
        if selections:
            scenario = st.selectbox("Select Scenario", ["Combined", "City", "Highway"])

            low, medium, high = controls.create_triplet(
                "distribution_plot",
                "Percentiles to display",
                0,
                100,
                10,
                50,
                90,
                1,
                ["Low", "Medium", "High"],
            )

            # Retrieve the simulation models from session_state
            models = st.session_state["simulation_models"]

            # Create one tab per model in the session
            tabs = st.tabs(list(models.keys()))
            for i, (model_name, model_obj) in enumerate(models.items()):
                with tabs[i]:
                    npv_array = model_obj.data["npv"][scenario]
                    if npv_array is None:
                        st.warning(
                            f"No NPV data available for {model_name} in {scenario}"
                        )
                        continue

                    # Sum across all forecast years to get a single total NPV per iteration
                    total_npv = npv_array.sum(axis=1)

                    # Create the two-subplot figure (PDF on top, CDF below)
                    fig = plotting.create_pdf_cdf_histogram(
                        total_npv, model_name, scenario, low, medium, high
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select models to create NPV distribution")

    @staticmethod
    def process_explanation(*args, **kwargs):
        """
        A new tab that provides a detailed recounting of how values are calculated
        in this app, from MPG-to-kWh/km conversion through Monte Carlo forecasting.
        """
        st.header("Process Explanation & Methodology")
        st.markdown(
            r"""
            ### 1. Converting MPG to kWh/km
            We start by taking taking the U.S. DOE figure for vehicle efficiency 
            \(see [www.fueleconomy.gov](https://www.fueleconomy.gov/feg/download.shtml)\) 
            *in miles per gallon (MPG)* 
            and convert it to *kilowatt-hours per kilometer (kWh/km)*.

            This conversion uses:
            $$
            \text{kWh per gallon} \approx 33.41
            $$

            $$
            \text{gallon per liter}  \approx 0.2199692
            $$

            $$
            \text{km per mile}  \approx 1.609344
            $$


            so that

            $$
            \text{kWh per km} = \frac{\text{kWh per gallon}}{\text{MPG} \times \text{km per mile} }
            $$
            ---

            ### 2. Monte Carlo Inputs & Distributions

            Each parameter (fuel price, distance traveled, interest rate, etc.) is modeled
            by a *Metalog distribution*, which is fitted to user-defined P10/P50/P90 values.
            When the simulation runs:
            - We **sample** from each distribution for each iteration and year.

            ---

            ### 3. Year-by-Year Cost Calculation

            For each vehicle model:

            1. **Energy Cost/Unit**: Based on the drawn electricity or gasoline price per kWh.
            2. **Distance**: Based on the sampled annual distance (km).
            3. **Efficiency**: The theoretical efficiency in kWh/km (possibly further adjusted by a performance factor).

            Hence the cost per year is calculated as:

            $$
            \text{Cost}_{year,i} \;=\; 
            \bigl(\text{kWh/km} \times \text{Price per kWh}\bigr) 
            \;\times\; \text{Distance}_{year,i}
            $$

            ---

            ### 4. Discounting with Annual Interest Rates

            We also sample an *annual interest rate* each iteration & year (1 + r). 
            We raise it to the power of the year index to form discount factors:

            $$
            \text{discount factor}_{year} \;=\; 
            (1 + r_i)^{\,\text{year}}
            $$

            Then the discounted cost in each year is:

            $$
            \text{NPV component}_{year,i} \;=\;
            \frac{\text{Cost}_{year,i}}{(1 + r_i)^{\,\text{year}}}
            $$

            ---

            ### 5. Summing NPV Across All Years

            For each iteration \(i\), we sum across the forecast horizon (years \(1\) through \(N_{\text{years}}\)):

            $$
            \text{NPV}_{i} \;=\;
            \sum_{\,\text{year}=1}^{N_{\text{years}}}\, 
            \text{NPV component}_{year,i}
            $$

            This yields a distribution of total NPVs across all Monte Carlo iterations.

            ---

            ### 6. Output & Visualization
            - **Forecasts Tab** plots percentile lines (P10, P50, P90) across each year.
            - **NPV Distribution Tab** shows how the final total NPV is distributed (PDF, CDF).
            - **NPV Sensitivity Tab** re-runs the forecast with each parameter “locked” to a min/max bound
              so we can see which parameter swings produce the biggest changes in NPV.

            ---
            **In summary**, this entire process—starting from the fundamental 
            MPG to kWh/km conversion, up to the discounted cash flow (NPV) calculations—enables 
            you to compare different vehicle options under uncertain future conditions.
            """
        )

    @staticmethod
    def dispatch(func_name: str, selections):
        """
        Dispatch method that calls one of the Layout's static methods by name.

        Args:
            func_name (str): The lowercased, underscored name of the tab function.
            selections (list): List of selected model IDs to pass to the function.

        Raises:
            st.error: If an invalid tab name is provided.
        """
        func = getattr(Layout, func_name, None)
        if func is not None:
            func(selections)
        else:
            st.error("Invalid tab name")

    @staticmethod
    def debug(selections, *args, **kwargs):
        """
        Utility method to inspect the Streamlit session state for debugging purposes.
        """
        st.write(st.session_state)


def create_tabs(selections):
    """
    Creates Streamlit tabs and dispatches to the relevant layout sections.
    Each tab corresponds to a method in the Layout class.

    Args:
        selections (list): Selected model IDs.

    Returns:
        None
    """
    names = [
        "Process Explanation",
        "Models",
        "Forecasts",
        "NPV Sensitivity",
        "NPV Distribution",
    ]

    tabs = st.tabs(names)

    for tab, tab_name in zip(tabs, names):
        with tab:
            func_name = tab_name.lower().replace(" ", "_")
            Layout.dispatch(func_name, selections)

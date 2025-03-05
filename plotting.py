from typing import Dict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import controls


np.float_ = np.float64


def create_dist_plot(m: str):
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


def create_forecast_plot():
    """
    Produces a 3×3 grid of subplots (for each selected model):
      Rows: 3 scenarios (City, Combined, Highway)
      Columns: 3 measures (Dollars per km, Dollars, Cumulative NPV)

    Each subplot shows P10/P50/P90 lines (with fill between P10 & P90)
    based on user-selected percentiles.
    """
    models = st.session_state.simulation_models
    low = 10
    medium = 50
    high = 90

    if not models:
        st.warning("No models found. Please select models first.")
        return

    # Create one Streamlit Tab per model
    tabs = st.tabs(list(models.keys()))

    # Define scenarios (each will be its own row)
    scenarios = ["City", "Combined", "Highway"]
    # Define measures (each will be its own column)
    # (measure_key, is_cumulative, display_label)
    measures = [
        ("dollars_per_km", False, "Dollars per km"),
        ("dollars", False, "Dollars"),
        ("npv", True, "Cumulative NPV"),
    ]

    scenario_colors = {
        "City": "rgba(0,255,0,{alpha})",
        "Combined": "rgba(255,0,0,{alpha})",
        "Highway": "rgba(0,0,255,{alpha})",
    }

    for i, (model_name, model_obj) in enumerate(models.items()):
        with tabs[i]:
            # Build the list of titles for each subplot: 3 rows × 3 columns
            subplot_titles = []
            for scenario in scenarios:
                for _, _, measure_label in measures:
                    subplot_titles.append(f"{scenario} – {measure_label}")

            # Create a 3×3 grid
            fig = make_subplots(
                rows=3,
                cols=3,
                shared_xaxes=False,
                horizontal_spacing=0.07,
                vertical_spacing=0.12,
                subplot_titles=subplot_titles,
            )
            color = scenario_colors[scenario]


            for row_idx, scenario in enumerate(scenarios, start=1):
                color = scenario_colors.get(scenario)
                for col_idx, (measure_key, is_cumulative, measure_label) in enumerate(
                    measures, start=1
                ):
                    # Grab the data array for this scenario & measure
                    data_array = model_obj.data[measure_key].get(scenario)
                    if data_array is None:
                        continue  # Might happen if scenario is missing

                    # Calculate percentile lines across iterations
                    p10_vals = np.percentile(data_array, low, axis=0)
                    p50_vals = np.percentile(data_array, medium, axis=0)
                    p90_vals = np.percentile(data_array, high, axis=0)

                    # Optionally cumulative
                    if is_cumulative:
                        p10_vals = np.cumsum(p10_vals)
                        p50_vals = np.cumsum(p50_vals)
                        p90_vals = np.cumsum(p90_vals)

                    x_vals = np.arange(data_array.shape[1])

                    # Fill between P10 and P90
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([p90_vals, p10_vals[::-1]]),
                            fill="toself",
                            fillcolor=color.format(alpha=0.1),
                            line=dict(color="rgba(255,255,255,0)"),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

                    # P10 line
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=p10_vals,
                            mode="lines",
                            line=dict(color=color.format(alpha=1), dash="dot"),
                            name=f"{scenario} P{low}",
                            # legendgroup=scenario,
                            showlegend=(col_idx == 1),
                            # Only show the legend entry once in the top-left subplot
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                    # P50 line
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=p50_vals,
                            mode="lines",
                            line=dict(color=color.format(alpha=1), dash="solid"),
                            name=f"{scenario} P{medium}",
                            # legendgroup=scenario,
                            showlegend=(col_idx == 1),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                    # P90 line
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=p90_vals,
                            mode="lines",
                            line=dict(color=color.format(alpha=1), dash="dash"),
                            name=f"{scenario} P{high}",
                            # legendgroup=scenario,
                            showlegend=(col_idx == 1),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

            # General styling
            fig.update_layout(
                # title=f"{model_name} – Forecast",
                # title_pad=dict(b=30),
                hovermode="x unified",
                height=1200,  # Might need extra height for 3 rows
                legend=dict(orientation="h", y=1.05),
                # margin=dict(t=100)
            )

            # Label axes. 3 rows × 3 columns => row/col each from 1 to 3
            for r in range(1, 4):
                for c in range(1, 4):
                    fig.update_xaxes(title_text="Year", row=r, col=c)

            # For Y-axis labels, you can be more explicit if you want
            # or you can leave them as is. For example:
            for c, (_, _, measure_label) in enumerate(measures, start=1):
                fig.update_yaxes(title_text=measure_label, row=1, col=c)
                # Then the rows below just re-use the same measure_label but you
                # might choose to set them blank to avoid repetition.

            st.plotly_chart(fig, use_container_width=True)


def create_tornado_figure(df, model_name="Tornado Chart", x_axis_title="NPV"):
    """
    Creates a 1×3 subplot figure where each column is a scenario (City, Combined, Highway).
    Each subplot is a 'tornado-style' sensitivity chart:
      - Horizontal lines from Low->Baseline and Baseline->High
      - Annotations for the Low and High numeric values
      - A baseline marker in the middle

    Args:
        df (pd.DataFrame): Must contain columns:
            ["Scenario", "Parameter", "Baseline", "Low", "High", "Lower Bound", "Upper Bound"]
            Each row describes how a single parameter changes the NPV under that scenario.
        model_name (str): Chart title for the entire figure.
        x_axis_title (str): Label for horizontal axis in each subplot (e.g. "NPV").
    """

    # 1) Identify the scenarios in your df (e.g. City, Combined, Highway).
    #    We define an order so that columns always appear as City, Combined, Highway.

    scenario_order = ["City", "Combined", "Highway"]
    available_scenarios = [s for s in scenario_order if s in df["Scenario"].unique()]

    # 2) Create a 1-row × N-col subplots layout (N=3 if all scenarios present)
    fig = make_subplots(
        rows=1,
        cols=len(available_scenarios),
        shared_yaxes=True,  # So parameters align across columns
        horizontal_spacing=0.08,  # Adjust spacing between columns
        subplot_titles=available_scenarios,
    )

    # 3) For each scenario, draw a horizontal bar from Low->Baseline and Baseline->High
    #    plus text annotations for the lower/upper numeric values.
    for col_idx, scenario in enumerate(available_scenarios, start=1):
        # Filter the df for just this scenario
        sub_df = df.loc[df["Scenario"] == scenario]
        sub_df.sort_values(
            ["Model", "Scenario", "Range"], ascending=False, inplace=True
        )

        # Sort parameters so we draw them in a consistent order (optional)
        # Often tornado charts are sorted by (High - Low). Example:
        # sub_df = sub_df.sort_values(by=["Range"], ascending=True)

        for _, row in sub_df.iterrows():
            param = row["Parameter"]
            baseline = row["Baseline"]
            low = row["Low"]
            high = row["High"]

            # 3A) Low→Baseline line
            fig.add_trace(
                go.Scatter(
                    x=[low, baseline],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="red", width=10),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            # Annotation at the "Low" end
            fig.add_annotation(
                x=low,
                y=param,
                text=f"{row['Lower Bound']}",  # the text to display
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                font=dict(size=10),
                row=1,
                col=col_idx,
            )

            # 3B) Baseline→High line
            fig.add_trace(
                go.Scatter(
                    x=[baseline, high],
                    y=[param, param],
                    mode="lines",
                    line=dict(color="green", width=10),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            # Annotation at the "High" end
            fig.add_annotation(
                x=high,
                y=param,
                text=f"{row['Upper Bound']}",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(size=10),
                row=1,
                col=col_idx,
            )

            # 3C) Baseline marker
            fig.add_trace(
                go.Scatter(
                    x=[baseline],
                    y=[param],
                    mode="markers",
                    marker=dict(color="black", size=12),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )

            fig.add_vline(
                x=baseline,
                line_dash="solid",
                line_color="grey",
                row=1,
                col=col_idx,
                annotation_text="P50",
                annotation_position="top",
                annotation_yshift=-15,
            )

        # Optional: Reverse the y-axis so highest-swing parameters appear at the top
        # or so parameters are read from top to bottom
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

    # 4) Adjust the overall layout
    #    - Increase the left margin if parameter names are long
    #    - Increase the top margin if subplot titles overlap
    fig.update_layout(
        # title=model_name,
        # The first scenario’s xaxis is "xaxis", second is "xaxis2", etc.
        # We can do a quick approach by updating all xaxes with same label:
        xaxis_title=x_axis_title,
        height=600,
        # margin=dict(l=150, r=50, t=80, b=50),
        hovermode="x unified",
        xaxis=dict(matches="x"),
        xaxis2=dict(matches="x"),
        xaxis3=dict(matches="x"),
    )

    # 5) Show the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def create_pdf_cdf_histogram():
    """
    Creates a 2×3 figure:
      - 2 rows: Top row is PDF (Histogram), bottom row is CDF (Cumulative Histogram)
      - 3 columns: One column per scenario (City, Combined, Highway)

    For each scenario, draws vertical lines at the user-chosen P10, P50, and P90.

    Args:
        scenarios_data (dict of str->np.array):
            A dict keyed by scenario name (e.g. {"City": <array>, "Combined": <array>, "Highway": <array>})
            Each value is an array of total NPV values for that scenario (one per MC iteration).
        model_name (str): Identifying label for the model, used in chart title.
        low (float): The first percentile to highlight (e.g. 10).
        medium (float): The second percentile to highlight (e.g. 50).
        high (float): The third percentile to highlight (e.g. 90).

    Returns:
        go.Figure: A Plotly figure with 2 rows and 3 columns of subplots (PDF & CDF x 3 scenarios).
    """

    # low, medium, high = controls.create_triplet(
    #     "distribution_plot",
    #     "Percentiles to display",
    #     0,
    #     100,
    #     10,
    #     50,
    #     90,
    #     1,
    #     ["Low", "Medium", "High"],
    # )
    low = 10
    medium = 50
    high = 90

    scenario_names = ["City", "Combined", "Highway"]

    models = st.session_state["simulation_models"]
    scenario_colors = {"City": "blue", "Combined": "green", "Highway": "red"}
    

    # Create one tab per model in the session
    tabs = st.tabs(list(models.keys()))
    for i, (model_name, model) in enumerate(models.items()):
        with tabs[i]:

            fig = make_subplots(
                rows=2,
                cols=3,
                shared_xaxes=False,
                horizontal_spacing=0.12,
                vertical_spacing=0.12,
            )

            # For each scenario, we place PDF in row=1, col=scenario_index
            # and CDF in row=2, col=scenario_index
            for col_idx, scenario in enumerate(scenario_names, start=1):

                total_npv = model.data["npv"][scenario].sum(axis=1)

                # PDF trace (top row)
                pdf_trace = go.Histogram(
                    x=total_npv,
                    histnorm="probability",
                    name=scenario,
                    marker=dict(color=scenario_colors[scenario]),
                    showlegend=False,
                    legendgroup=scenario,
                )
                fig.add_trace(pdf_trace, row=1, col=col_idx)

                # CDF trace (bottom row)
                cdf_trace = go.Histogram(
                    x=total_npv,
                    histnorm="probability",
                    cumulative_enabled=True,
                    name=scenario,
                    marker=dict(color=scenario_colors[scenario]),
                    showlegend=True,
                    legendgroup=scenario,
                )
                fig.add_trace(cdf_trace, row=2, col=col_idx)

                # Calculate P10, P50, P90
                p10_val = np.percentile(total_npv, low)
                p50_val = np.percentile(total_npv, medium)
                p90_val = np.percentile(total_npv, high)

                # Add vertical lines at P10, P50, P90 in both subplots (PDF & CDF)
                # We'll put the label in the PDF row only (row=1).
                for row_idx in [1, 2]:
                    # P10
                    fig.add_vline(
                        x=p10_val,
                        line_dash="dot",
                        line_color="grey",
                        row=row_idx,
                        col=col_idx,
                        annotation_text=("P" + str(int(low))) if row_idx == 1 else "",
                        annotation_position="top",
                    )
                    # P50
                    fig.add_vline(
                        x=p50_val,
                        line_dash="solid",
                        line_color="grey",
                        row=row_idx,
                        col=col_idx,
                        annotation_text=(
                            ("P" + str(int(medium))) if row_idx == 1 else ""
                        ),
                        annotation_position="top",
                    )
                    # P90
                    fig.add_vline(
                        x=p90_val,
                        line_dash="dash",
                        line_color="grey",
                        row=row_idx,
                        col=col_idx,
                        annotation_text=("P" + str(int(high))) if row_idx == 1 else "",
                        annotation_position="top",
                    )

            # Style & labeling
            fig.update_xaxes(
                title_text="NPV", row=2, col=1
            )  # you can do more row/col combos if you want
            fig.update_yaxes(title_text="Density (PDF)", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

            fig.update_layout(
                # title=f"{model_name} – PDF & CDF Histograms for All Scenarios",
                hovermode="x unified",
                height=700,
                legend=dict(orientation="h", y=1.10),
                # title_pad=dict(b=20),
                # If you'd like to see all subplots share x-range, you can do:
                xaxis=dict(matches="x"),
                xaxis2=dict(matches="x"),
                xaxis3=dict(matches="x"),
                xaxis4=dict(matches="x"),
                xaxis5=dict(matches="x"),
                xaxis6=dict(matches="x"),
            )
            st.plotly_chart(fig, use_container_width=True)

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
    Creates an enhanced forecast plot with:
      1) Confidence bands (P10–P90) for each scenario
      2) Subplots for each scenario (City, Combined, Highway), each with two columns:
         left column for non-cumulative, right column for cumulative
      3) A slider to control how many forecast years to display
      4) A complete legend for all percentile lines
      5) One figure per model, displayed in separate Streamlit tabs.

    Args:
        measure (str): One of "dollars_per_km", "dollars", or "npv".
    """
    models = st.session_state.simulation_models

    measure = st.selectbox("Select Measure", ["Dollars Per km", "Dollars", "NPV"])

    low, medium, high = controls.create_triplet(
        "forecast_plot",
        "Percentiles to display",
        0,
        100,
        10,
        50,
        90,
        1,
        ["Low", "Medium", "High"],
    )

    measure = measure.lower().replace(" ", "_")

    if not models:
        st.warning("No models found. Please select models first.")
        return

    # We create a tab for each model
    tabs = st.tabs(list(models.keys()))
    scenarios = ["City", "Combined", "Highway"]  # or dynamically detect from data keys

    for i, (model_name, model_obj) in enumerate(models.items()):
        with tabs[i]:
            # We have 3 scenarios and 2 columns => 3*2 = 6 subplots total
            # We'll build the list of subplot titles so it doesn't throw a ValueError
            subplot_titles = []
            for scenario in scenarios:
                subplot_titles.append(f"{scenario} - Non-Cumulative")
                subplot_titles.append(f"{scenario} - Cumulative")

            # Build subplots: 3 rows, 2 columns
            fig = make_subplots(
                rows=len(scenarios),
                cols=2,
                shared_xaxes=False,
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
                subplot_titles=subplot_titles,
            )

            # We'll store a color for each scenario so lines have separate colors
            scenario_colors = {"City": "blue", "Combined": "green", "Highway": "red"}

            row_index = 1
            for scenario in scenarios:
                data_array = model_obj.data[measure].get(scenario)

                # Compute percentile lines
                p10 = np.percentile(data_array, low, axis=0)
                p50 = np.percentile(data_array, medium, axis=0)
                p90 = np.percentile(data_array, high, axis=0)

                # Build x-values (year indices)
                x_vals = np.arange(data_array.shape[1])
                color = scenario_colors.get(
                    scenario, "blue"
                )  # fallback color if missing

                def add_scenario_traces(
                    x, p10_, p50_, p90_, row, col, cumulative=False
                ):
                    # Optionally make them cumulative
                    if cumulative:
                        p10_ = np.cumsum(p10_)
                        p50_ = np.cumsum(p50_)
                        p90_ = np.cumsum(p90_)

                    # Confidence band (fill area between p10 & p90)
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x, x[::-1]]),
                            y=np.concatenate([p90_, p10_[::-1]]),
                            fill="toself",
                            fillcolor="rgba(255, 255, 255, 0.1)",  # Light gray shade
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,  # no extra legend
                            legendgroup=f"{scenario} {'(cum)' if cumulative else ''}",
                        ),
                        row=row,
                        col=col,
                    )

                    # P10 line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=p10_,
                            mode="lines",
                            line=dict(color=color, dash="dot"),
                            name=f"{scenario} P{low} {'(cum)' if cumulative else ''}",
                            showlegend=True,
                            legendgroup=f"{scenario} {'(cum)' if cumulative else ''}",
                        ),
                        row=row,
                        col=col,
                    )

                    # P50 line (Median)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=p50_,
                            mode="lines",
                            line=dict(color=color, dash="solid"),
                            name=f"{scenario} P{medium} {'(cum)' if cumulative else ''}",
                            showlegend=True,
                            legendgroup=f"{scenario} {'(cum)' if cumulative else ''}",
                        ),
                        row=row,
                        col=col,
                    )

                    # P90 line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=p90_,
                            mode="lines",
                            line=dict(color=color, dash="dash"),
                            name=f"{scenario} P{high} {'(cum)' if cumulative else ''}",
                            showlegend=True,
                            legendgroup=f"{scenario} {'(cum)' if cumulative else ''}",
                        ),
                        row=row,
                        col=col,
                    )

                # Left column => Non-cumulative
                add_scenario_traces(
                    x_vals, p10, p50, p90, row_index, 1, cumulative=False
                )
                # Right column => Cumulative
                add_scenario_traces(
                    x_vals, p10, p50, p90, row_index, 2, cumulative=True
                )

                row_index += 1

            fig.update_layout(
                title=f"{model_name} – {measure} Forecast",
                hovermode="x unified",
                height=900,
            )

            # Label axes
            # For each row, columns 1 and 2
            for r in range(1, len(scenarios) + 1):
                fig.update_xaxes(title_text="Year", row=r, col=1)
                fig.update_xaxes(title_text="Year", row=r, col=2)
                fig.update_yaxes(title_text="Cost", row=r, col=1)
                fig.update_yaxes(title_text="Cost", row=r, col=2)

            st.plotly_chart(fig, use_container_width=True)


def create_tornado_figure(df: pd.DataFrame, title, x_axis_title="NPV"):
    """
    Creates a tornado diagram (sensitivity chart) from a DataFrame where each row
    includes:
      - 'Parameter': the parameter name
      - 'Baseline': baseline scenario
      - 'Low': NPV at parameter's low bound
      - 'High': NPV at parameter's high bound
      - 'low_color': color for the segment from Low to Baseline
      - 'high_color': color for the segment from Baseline to High

    The horizontal bars represent how varying a parameter from its Low to High bound
    affects the model's NPV.

    Args:
        df (pd.DataFrame): Data must contain columns:
                           ['Parameter', 'Baseline', 'Low', 'High', 'low_color', 'high_color'].
        title (str): Chart title, often the model name or something describing the scenario.
        x_axis_title (str): Label for the horizontal axis, typically "NPV".
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        param = row["Parameter"]
        baseline = row["Baseline"]
        low = row["Low"]
        high = row["High"]

        fig.add_trace(
            go.Scatter(
                x=[low, baseline],
                y=[param, param],
                mode="lines",
                line=dict(color="red", width=10),
                showlegend=False,
            )
        )

        fig.add_annotation(
            x=low,
            y=param,
            text=f"{row['Lower Bound']}",
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            font=dict(size=10),
        )

        fig.add_trace(
            go.Scatter(
                x=[baseline, high],
                y=[param, param],
                mode="lines",
                line=dict(color="green", width=10),
                showlegend=False,
                # name=row["Parameter High"],
                name="",
            )
        )

        fig.add_annotation(
            x=high,
            y=param,
            text=f"{row['Upper Bound']}",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(size=10),
        )

    # Marker for baseline
    fig.add_trace(
        go.Scatter(
            x=[baseline],
            y=[param],
            mode="markers",
            marker=dict(color="black", size=12),
            showlegend=False,
            hoverinfo="skip",
            name="",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis=dict(title="Parameter", automargin=True),
        margin=dict(l=150, r=50, t=80, b=50),
        height=600,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def create_pdf_cdf_histogram(total_npv, model_name, scenario, low, medium, high):
    """
    Creates a figure with two stacked histograms (2 rows, 1 column):
      1. Top row: a histogram of total NPV with histnorm='probability' (PDF).
      2. Bottom row: the same data but cumulative=True (CDF).
    Additionally, adds vertical dashed lines at the P10, P50, and P90
    values in both subplots.

    Args:
        total_npv (array-like): An array of total NPV values (one per Monte Carlo iteration).
        model_name (str): The name/ID of the model for the chart title.
        scenario (str): The driving scenario (City, Combined, Highway) for the chart title.

    Returns:
        go.Figure: The final Plotly figure with two subplots.
    """

    # Calculate P10, P50, P90
    p10_val = np.percentile(total_npv, low)
    p50_val = np.percentile(total_npv, medium)
    p90_val = np.percentile(total_npv, high)

    # Create two-row subplot layout with shared x-axes
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["PDF (Histogram)", "CDF (Cumulative Histogram)"],
    )

    # --- PDF histogram (top row) ---
    pdf_trace = go.Histogram(
        x=total_npv, histnorm="probability", name="NPV PDF"  # PDF-like histogram
    )
    fig.add_trace(pdf_trace, row=1, col=1)

    # --- CDF histogram (bottom row) ---
    cdf_trace = go.Histogram(
        x=total_npv,
        histnorm="probability",
        cumulative_enabled=True,  # Turn on cumulative mode
        name="NPV CDF",
    )
    fig.add_trace(cdf_trace, row=2, col=1)

    # Add vertical lines for P10, P50, P90 on both subplots
    quantiles = [(p10_val, f"P{low}"), (p50_val, f"P{medium}"), (p90_val, f"P{high}")]

    # We'll position P10 label on the left, P50 center, P90 right
    # so they don't overlap each other horizontally
    annotation_positions = ["left", "center", "right"]

    for subplot_index, row_label in [(1, "top"), (2, "bottom")]:
        for (x_val, label), pos in zip(quantiles, annotation_positions):
            fig.add_vline(
                x=x_val,
                line_dash="dash",
                line_color="black",
                annotation_text=label,
                # annotation_position=f"{row_label} {pos}",
                row=subplot_index,
                col=1,
            )

    # Update axis labels
    fig.update_xaxes(title_text="NPV", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

    # Layout adjustments
    fig.update_layout(
        title=f"{model_name} - {scenario} - NPV PDF & CDF",
        hovermode="x unified",
        height=700,
    )
    return fig

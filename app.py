import streamlit as st
import layout


st.set_page_config(page_title="Vehicle Operating Cost Forecast", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state defaults if they haven't been set
if "years" not in st.session_state:
    st.session_state.years = 10

if "iterations" not in st.session_state:
    st.session_state.iterations = 1000

if "simulation_models" not in st.session_state:
    st.session_state.simulation_models = {}

if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

if "distribution_data" not in st.session_state:
    st.session_state.distribution_data = {}


with st.sidebar:
    # 
    # Renders the controls layout in the sidebar, which lets the user select models
    # and adjust distributions and parameters for the forecast.
    # 
    selections = layout.Layout.controls()

# 
# Main content area that creates the various tabs (Models, Forecasts, NPV Sensitivity, etc.)
# based on user selections, and dispatches to the appropriate methods in layout.py
# 
layout.create_tabs(selections)

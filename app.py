import streamlit as st
import layout
from models import Simulation


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


if "simulation" not in st.session_state:
    st.session_state.simulation = Simulation()


with st.sidebar:
    # 
    # Renders the controls layout in the sidebar, which lets the user select models
    # and adjust distributions and parameters for the forecast.
    # 
    layout.Layout.controls()

# 
# Main content area that creates the various tabs (Models, Forecasts, NPV Sensitivity, etc.)
# based on user selections, and dispatches to the appropriate methods in layout.py
# 
layout.create_tabs()

"""
data module


"""

import pandas as pd
import yaml
import streamlit as st

models = pd.read_csv("data/models.csv")


def get_config():
    with open("data/config.yaml", "r") as f:
        return yaml.safe_load(f)

def get(scenario:str,dataset:str):

    return pd.read_csv(f"data/{scenario}_{dataset}.csv")

def get_bounds():

    config = get_config()

    bounds = []

    for distribution in config['distributions']:
        bounds.append(
            {
                'Parameter': distribution['label'],
                'Lower Bound': distribution['min_value'],
                'Upper Bound': distribution['max_value'],
            }
        )

    return pd.DataFrame(bounds)




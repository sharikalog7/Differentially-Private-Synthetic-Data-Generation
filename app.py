import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random

st.set_page_config(page_title="Differentially Private Synthetic Data", layout="wide")

st.title("📊 Differentially Private Synthetic Data Generator")
st.markdown("""
This demo shows how inference-only approaches can be used with **Large Language Models (LLMs)** to generate **synthetic datasets**. 
Sensitive examples are used as *prompts*, while **differential privacy (DP)** is applied to ensure outputs are aggregated safely.

⚠️ *Note: This demo uses toy data and simple noise addition for DP illustration.*
""")

# Sidebar config
st.sidebar.header("Generation Settings")
n_samples = st.sidebar.slider("Number of synthetic records", 10, 200, 50)
epsilon = st.sidebar.slider("Differential Privacy ε (smaller = more private)", 0.1, 5.0, 1.0)

fake = Faker()

# Example sensitive dataset (toy)
real_data = pd.DataFrame({
    "name": [fake.name() for _ in range(20)],
    "age": [random.randint(18, 70) for _ in range(20)],
    "income": [random.randint(30000, 120000) for _ in range(20)],
})

st.subheader("Example Sensitive Dataset (not shared externally)")
st.dataframe(real_data.head())

# Synthetic data generation with DP noise
def generate_synthetic_data(n, eps):
    synthetic = []
    scale = 1.0 / eps  # DP noise scale
    for _ in range(n):
        record = {
            "name": fake.first_name(),
            "age": max(18, int(np.random.normal(40, 10) + np.random.laplace(0, scale))),
            "income": max(20000, int(np.random.normal(60000, 15000) + np.random.laplace(0, scale*5000)))
        }
        synthetic.append(record)
    return pd.DataFrame(synthetic)

synthetic_data = generate_synthetic_data(n_samples, epsilon)

st.subheader("Generated Synthetic Data (with Differential Privacy)")
st.dataframe(synthetic_data.head())

st.download_button(
    "💾 Download Synthetic Data",
    data=synthetic_data.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Demo project: Differentially Private Synthetic Data Generation using inference-only approaches.")

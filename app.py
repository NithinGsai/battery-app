import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go
import requests

st.set_page_config(layout="wide")
st.title("Battery App")

# ================= MODEL DOWNLOAD =================
MODEL_PATH = "battery_performance_models.joblib"
MODEL_URL = "https://drive.google.com/uc?id=1lb1X18didE43TcUSYCyZZPFuovmiNw5R"


@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run)..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return joblib.load(MODEL_PATH)


models = load_models()

# ================= SESSION =================
if "ran_once" not in st.session_state:
    st.session_state.ran_once = False
if "tables" not in st.session_state:
    st.session_state.tables = None
if "curves" not in st.session_state:
    st.session_state.curves = None

# ================= INPUTS =================
t_peak = st.sidebar.slider("T_Peak (°C)", 25, 80, 55)
x_hs_mm = st.sidebar.slider("x_hs (mm)", 0.0, 40.0, 20.0)

radius_mm = 10.5
yc_mm = 30.0

y_hs_mm = st.sidebar.slider("y_hs (mm)", yc_mm - radius_mm, yc_mm + radius_mm, yc_mm)
z_hs_mm = st.sidebar.slider("z_hs (mm)", 20.0, 60.0, 50.0)
soc = st.sidebar.slider("SOC", 0.2, 1.0, 0.8)

# convert to meters
x_hs = x_hs_mm / 1000
y_hs = y_hs_mm / 1000
z_hs = z_hs_mm / 1000

x_model = 0.04 - x_hs if x_hs < 0.02 else x_hs

# ================= 3D PANEL =================
with st.container(border=True):

    fig = go.Figure()

    radius = 0.0105
    centers = [0.0, 0.02, 0.04]
    yc = 0.03
    zmin, zmax = 0.02, 0.08

    theta = np.linspace(0, 2*np.pi, 40)
    z_vals = np.linspace(zmin, zmax, 40)
    theta, z_vals = np.meshgrid(theta, z_vals)

    for xc in centers:
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)

        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z_vals,
            opacity=0.4,
            showscale=False
        ))

    # hotspot
    fig.add_trace(go.Scatter3d(
        x=[x_hs],
        y=[y_hs],
        z=[z_hs],
        mode="markers",
        marker=dict(size=6, color="red")
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(dragmode="orbit")
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displayModeBar": False
        }
    )

# ================= COMPUTE =================
compute_btn = st.sidebar.button("Compute")

run_compute = False
if not st.session_state.ran_once:
    run_compute = True
    st.session_state.ran_once = True
if compute_btn:
    run_compute = True

if run_compute and models is not None:

    soc_points = np.linspace(1.0, 0.2, 5)

    volt_rows = []
    temp_rows = []

    for s in soc_points:

        df = pd.DataFrame([{
            "T_Peak": t_peak,
            "x_hs": x_model,
            "y_hs": y_hs,
            "z_hs": z_hs,
            "SOC": s
        }])

        pred = {name: model.predict(df)[0] for name, model in models.items()}

        volt_rows.append({
            "SOC": round(s, 2),
            "Cell1": pred.get("Cell1 (V)", 0),
            "Cell2": pred.get("Cell2 (V)", 0),
            "Cell3": pred.get("Cell3 (V)", 0)
        })

        temp_rows.append({
            "SOC": round(s, 2),
            "Cell1": pred.get("T_cell1", 0),
            "Cell2": pred.get("T_cell2", 0),
            "Cell3": pred.get("T_cell3", 0)
        })

    st.session_state.tables = (
        pd.DataFrame(volt_rows),
        pd.DataFrame(temp_rows)
    )

    # curves
    soc_vals = np.linspace(1.0, 0.2, 25)

    temp = {"SOC": []}
    volt = {"SOC": []}

    for s in soc_vals:
        temp["SOC"].append(s)
        volt["SOC"].append(s)

        df2 = pd.DataFrame([{
            "T_Peak": t_peak,
            "x_hs": x_model,
            "y_hs": y_hs,
            "z_hs": z_hs,
            "SOC": s
        }])

        for name, model in models.items():
            val = model.predict(df2)[0]

            if "T_cell" in name:
                temp.setdefault(name, []).append(val)

            if "(V)" in name:
                volt.setdefault(name, []).append(val)

    st.session_state.curves = (pd.DataFrame(temp), pd.DataFrame(volt))

# ================= DISPLAY =================
if st.session_state.tables and st.session_state.curves:

    volt_table, temp_table = st.session_state.tables
    temp_df, volt_df = st.session_state.curves

    # TEMPERATURE
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Temperature Table")
        st.dataframe(temp_table)

    with col2:
        st.subheader("Temperature vs SOC")
        fig1 = go.Figure()
        for col in temp_df.columns:
            if col != "SOC":
                fig1.add_trace(go.Scatter(
                    x=temp_df["SOC"],
                    y=temp_df[col],
                    mode="lines",
                    name=col
                ))
        fig1.update_layout(xaxis=dict(autorange="reversed"))
        st.plotly_chart(fig1, use_container_width=True)

    # VOLTAGE
    col3, col4 = st.columns([1, 2])

    with col3:
        st.subheader("Voltage Table")
        st.dataframe(volt_table)

    with col4:
        st.subheader("Voltage vs SOC")
        fig2 = go.Figure()
        for col in volt_df.columns:
            if col != "SOC":
                fig2.add_trace(go.Scatter(
                    x=volt_df["SOC"],
                    y=volt_df[col],
                    mode="lines",
                    name=col
                ))
        fig2.update_layout(xaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)
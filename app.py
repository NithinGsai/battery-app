import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go
import requests

st.set_page_config(layout="wide")
st.title("Battery App")

# ================= MODEL =================
MODEL_PATH = "battery_performance_models.joblib"
MODEL_URL = "https://drive.google.com/uc?id=1lb1X18didE43TcUSYCyZZPFuovmiNw5R"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(MODEL_PATH)

models = load_models()

# ================= INPUTS =================
t_peak = st.sidebar.slider("T_Peak (°C)", 25, 80, 55)
x_hs_mm = st.sidebar.slider("x_hs (mm)", 0.0, 40.0, 20.0)

radius_mm = 10.5
yc_mm = 30.0

y_hs_mm = st.sidebar.slider("y_hs (mm)", yc_mm - radius_mm, yc_mm + radius_mm, yc_mm)
z_hs_mm = st.sidebar.slider("z_hs (mm)", 20.0, 60.0, 50.0)
soc = st.sidebar.slider("SOC", 0.2, 1.0, 0.8)

# convert
x_hs = x_hs_mm / 1000
y_hs = y_hs_mm / 1000
z_hs = z_hs_mm / 1000

# ================= MIRROR =================
mirrored = x_hs < 0.02
x_model = 0.04 - x_hs if mirrored else x_hs

# ================= 3D =================
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
    fig.add_trace(go.Surface(x=x, y=y, z=z_vals, opacity=0.4, showscale=False))

fig.add_trace(go.Scatter3d(
    x=[x_hs],
    y=[y_hs],
    z=[z_hs],
    mode="markers",
    marker=dict(size=6, color="red")
))

fig.update_layout(height=500)
st.plotly_chart(fig, config={"scrollZoom": False})

# ================= COMPUTE ALWAYS =================
if models is not None:

    # ================= EXACT OUTPUT =================
    st.subheader("Exact Output at Current Input")

    df_current = pd.DataFrame([{
        "T_Peak": t_peak,
        "x_hs": x_model,
        "y_hs": y_hs,
        "z_hs": z_hs,
        "SOC": soc
    }])

    pred_current = {name: model.predict(df_current)[0] for name, model in models.items()}

    if mirrored:
        v1, v3 = pred_current["Cell3 (V)"], pred_current["Cell1 (V)"]
        t1, t3 = pred_current["T_cell3"], pred_current["T_cell1"]
    else:
        v1, v3 = pred_current["Cell1 (V)"], pred_current["Cell3 (V)"]
        t1, t3 = pred_current["T_cell1"], pred_current["T_cell3"]

    v2 = pred_current["Cell2 (V)"]
    t2 = pred_current["T_cell2"]

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Voltage (V)")
        st.metric("Cell1", f"{v1:.4f}")
        st.metric("Cell2", f"{v2:.4f}")
        st.metric("Cell3", f"{v3:.4f}")

    with colB:
        st.subheader("Temperature (°C)")
        st.metric("Cell1", f"{t1:.2f}")
        st.metric("Cell2", f"{t2:.2f}")
        st.metric("Cell3", f"{t3:.2f}")

    # ================= TABLES =================
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

        if mirrored:
            c1_v, c3_v = pred["Cell3 (V)"], pred["Cell1 (V)"]
            c1_t, c3_t = pred["T_cell3"], pred["T_cell1"]
        else:
            c1_v, c3_v = pred["Cell1 (V)"], pred["Cell3 (V)"]
            c1_t, c3_t = pred["T_cell1"], pred["T_cell3"]

        volt_rows.append({
            "SOC": round(s, 2),
            "Cell1": c1_v,
            "Cell2": pred["Cell2 (V)"],
            "Cell3": c3_v
        })

        temp_rows.append({
            "SOC": round(s, 2),
            "Cell1": c1_t,
            "Cell2": pred["T_cell2"],
            "Cell3": c3_t
        })

    volt_table = pd.DataFrame(volt_rows)
    temp_table = pd.DataFrame(temp_rows)

    # ================= CURVES =================
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

        pred = {name: model.predict(df2)[0] for name, model in models.items()}

        if mirrored:
            temp.setdefault("T_cell1", []).append(pred["T_cell3"])
            temp.setdefault("T_cell2", []).append(pred["T_cell2"])
            temp.setdefault("T_cell3", []).append(pred["T_cell1"])

            volt.setdefault("Cell1 (V)", []).append(pred["Cell3 (V)"])
            volt.setdefault("Cell2 (V)", []).append(pred["Cell2 (V)"])
            volt.setdefault("Cell3 (V)", []).append(pred["Cell1 (V)"])
        else:
            temp.setdefault("T_cell1", []).append(pred["T_cell1"])
            temp.setdefault("T_cell2", []).append(pred["T_cell2"])
            temp.setdefault("T_cell3", []).append(pred["T_cell3"])

            volt.setdefault("Cell1 (V)", []).append(pred["Cell1 (V)"])
            volt.setdefault("Cell2 (V)", []).append(pred["Cell2 (V)"])
            volt.setdefault("Cell3 (V)", []).append(pred["Cell3 (V)"])

    temp_df = pd.DataFrame(temp)
    volt_df = pd.DataFrame(volt)

    # ================= DISPLAY =================
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
        st.plotly_chart(fig1)

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
        st.plotly_chart(fig2)
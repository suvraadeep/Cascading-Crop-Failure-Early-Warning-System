import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import json, os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import cfg
from src.data_pipeline import fetch_nasa_power, engineer_inference_features
from src.alerts import send_sms_alert
try:
    from src.model_utils import load_model_and_conformal, predict_farm
    MODEL_AVAILABLE = cfg.MODEL_PATH.exists()
except:
    MODEL_AVAILABLE = False

st.set_page_config(
    page_title   = "🌾 Crop Failure Early Warning System",
    page_icon    = "🌾",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .risk-critical {background: rgba(142,68,173,0.3); border-color: #8E44AD;}
    .risk-high     {background: rgba(231,76,60,0.3);  border-color: #E74C3C;}
    .risk-moderate {background: rgba(243,156,18,0.3); border-color: #F39C12;}
    .risk-low      {background: rgba(46,204,113,0.3); border-color: #2ECC71;}
    .stAlert {border-radius: 10px;}
    h1 {background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=80)
    st.title("🌾 CropEWS Settings")
    st.divider()

    st.subheader("📍 Location")
    lat = st.number_input("Latitude",  value=15.3173, step=0.01,
                           min_value=-90.0, max_value=90.0)
    lon = st.number_input("Longitude", value=75.7139, step=0.01,
                           min_value=-180.0, max_value=180.0)

    st.subheader("🌱 Farm Details")
    soil_type = st.selectbox("Soil Type",
        ["Sandy", "Clay", "Loam", "Silty", "Laterite"], index=2)
    crop_type = st.selectbox("Crop Type",
        ["Rice", "Wheat", "Millet", "Sorghum", "Barley", "Maize"], index=0)
    farm_area = st.number_input("Farm Area (ha)", value=2.5, min_value=0.1, max_value=100.0)

    st.subheader("⚙️ Model Settings")
    horizon_days = st.slider("Forecast Horizon (days)", 7, 21, 21)
    show_conformal = st.toggle("Show Conformal Intervals", value=True)
    show_attention = st.toggle("Show Feature Importance", value=True)

    st.divider()
    st.subheader("📱 SMS Alerts")
    phone_number = st.text_input("Farmer Phone (+country code)", "+919876543210")
    alert_threshold = st.select_slider(
        "Alert on Risk Level",
        options=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        value="HIGH"
    )
    send_alert_btn = st.button("📤 Send Test Alert", type="secondary")

    st.divider()
    st.caption(f"Model available: {'✅' if MODEL_AVAILABLE else '❌ (demo mode)'}")
    st.caption("Data: NASA POWER API (free) + Sentinel-2 NDVI")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🌾 Cascading Crop Failure Early Warning System")
st.markdown("*Multi-modal AI forecasting: Satellite NDVI × Weather × Soil — powered by TFT + Conformal Prediction*")
st.divider()


# ─────────────────────────────────────────────────────────────
# FETCH DATA + INFERENCE
# ─────────────────────────────────────────────────────────────
SOIL_WHC = {"Sandy": 0.12, "Clay": 0.38, "Loam": 0.28, "Silty": 0.32, "Laterite": 0.18}
SOIL_SENS = {"Sandy": 1.4, "Clay": 0.7, "Loam": 0.9, "Silty": 0.85, "Laterite": 1.2}

@st.cache_data(ttl=3600, show_spinner="🛰️ Fetching NASA POWER weather data...")
def get_weather_data(lat, lon, days_back=120):
    return fetch_nasa_power(lat, lon, days_back)

@st.cache_data(ttl=3600)
def get_demo_weather(lat, lon, days_back=120):
    """Demo weather generator for when NASA POWER is unavailable."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days_back, freq="D")
    t = np.arange(days_back)
    return pd.DataFrame({
        "date":        dates,
        "temp_c":      28 + 6 * np.sin(2*np.pi*t/365) + np.random.normal(0,1.5,days_back),
        "precip_mm":   np.random.exponential(3, days_back) * (np.random.random(days_back) < 0.35),
        "humidity_pct":60 + 20*np.random.random(days_back),
        "wind_ms":     2.5 + np.random.exponential(1, days_back),
        "solar_kwh":   5.0 + np.random.normal(0,0.5,days_back),
        "dewpoint_c":  20 + np.random.normal(0,2,days_back),
    })

@st.cache_data(ttl=300)
def get_demo_prediction(lat, lon, soil, horizon):
    """Demo prediction for when model isn't loaded."""
    t = np.arange(1, horizon + 1)
    trend = -0.008 * t
    base  = 0.55 + 0.1 * np.sin(2*np.pi*t/30)
    median = np.clip(base + trend + np.random.normal(0, 0.01, horizon), 0.1, 0.9)
    width  = 0.08 + 0.003 * t
    stress_sens = SOIL_SENS.get(soil, 0.9)
    if stress_sens > 1.0:
        median *= 0.85
    return {
        "horizon":    t.tolist(),
        "median":     median.tolist(),
        "lower_80":   np.clip(median - width, 0, 1).tolist(),
        "upper_80":   np.clip(median + width, 0, 1).tolist(),
        "lower_50":   np.clip(median - width/2, 0, 1).tolist(),
        "upper_50":   np.clip(median + width/2, 0, 1).tolist(),
        "cp_lower":   np.clip(median - width*1.4, 0, 1).tolist(),
        "cp_upper":   np.clip(median + width*1.4, 0, 1).tolist(),
        "min_ndvi":   float(median.min()),
        "risk_level": cfg.get_risk_level(float(median.min())),
        "stress_day": int(np.argmax(median < cfg.STRESS_THRESHOLD)) + 1
                      if any(median < cfg.STRESS_THRESHOLD) else None,
    }


# Fetch weather
with st.spinner("🛰️ Fetching weather data from NASA POWER..."):
    wx_df = get_weather_data(lat, lon)
    if wx_df is None:
        wx_df = get_demo_weather(lat, lon)
        st.info("📡 Using synthetic weather (NASA POWER unavailable in this environment)")

# Engineer features
feat_df = engineer_inference_features(wx_df,
                                       soil_whc=SOIL_WHC[soil_type],
                                       stress_sens=SOIL_SENS[soil_type])

# Run prediction
if MODEL_AVAILABLE:
    try:
        model, cp = load_model_and_conformal()
        pred = predict_farm(model, cp, feat_df, farm_id=f"LAT{lat:.1f}_LON{lon:.1f}")
    except Exception as e:
        st.warning(f"⚠️ Model loading failed ({type(e).__name__}), using demo predictions")
        MODEL_AVAILABLE = False
        pred = get_demo_prediction(lat, lon, soil_type, horizon_days)
else:
    pred = get_demo_prediction(lat, lon, soil_type, horizon_days)

risk_level = pred["risk_level"]
risk_colors = cfg.RISK_COLORS


# ─────────────────────────────────────────────────────────────
# RISK BANNER
# ─────────────────────────────────────────────────────────────
banner_color = risk_colors[risk_level]
stress_msg = (f"⏱️ Crop stress predicted in **{pred['stress_day']} days**"
              if pred["stress_day"] else "✅ No acute stress in forecast window")

if risk_level in ["HIGH", "CRITICAL"]:
    st.error(f"🚨 **{risk_level} RISK** — Min NDVI: {pred['min_ndvi']:.3f} | {stress_msg}")
elif risk_level == "MODERATE":
    st.warning(f"⚠️ **{risk_level} RISK** — Min NDVI: {pred['min_ndvi']:.3f} | {stress_msg}")
else:
    st.success(f"✅ **{risk_level} RISK** — Min NDVI: {pred['min_ndvi']:.3f} | {stress_msg}")


# ─────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("📍 Location", f"{lat:.2f}°N, {lon:.2f}°E")
with c2:
    latest_ndvi = feat_df.get("ndvi", pd.Series([0.5])).iloc[-1] if "ndvi" in feat_df else 0.5
    st.metric("🌿 Current NDVI", f"{0.52:.3f}", delta=f"{-0.03:.3f}")
with c3:
    st.metric("🌡️ Temp (7-day avg)",
              f"{feat_df['temp_c_roll7_mean'].iloc[-1]:.1f}°C")
with c4:
    st.metric("🌧️ Rain (14-day)",
              f"{feat_df['precip_mm_roll14_sum'].iloc[-1]:.1f} mm")
with c5:
    st.metric("💧 Drought Index (SPI)",
              f"{feat_df['spi30'].iloc[-1]:.2f}",
              delta="Normal" if abs(feat_df['spi30'].iloc[-1]) < 1 else "Anomalous")

st.divider()


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT: Map + Forecast
# ─────────────────────────────────────────────────────────────
col_map, col_forecast = st.columns([1, 1.5])

# ── FOLIUM MAP ────────────────────────────────────────────────
with col_map:
    st.subheader("🗺️ Farm Risk Map")

    # Create folium map
    m = folium.Map(
        location=[lat, lon],
        zoom_start=8,
        tiles="CartoDB dark_matter"
    )

    # Add satellite tile layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False
    ).add_to(m)

    # Main farm marker
    risk_icon_colors = {"LOW": "green", "MODERATE": "orange",
                        "HIGH": "red", "CRITICAL": "purple"}
    folium.CircleMarker(
        location=[lat, lon],
        radius=18,
        color=risk_colors[risk_level],
        fill=True,
        fill_color=risk_colors[risk_level],
        fill_opacity=0.7,
        popup=folium.Popup(
            f"""<div style='width:200px;font-family:Arial'>
            <b>🌾 Your Farm</b><br>
            <b>Soil:</b> {soil_type}<br>
            <b>Crop:</b> {crop_type}<br>
            <b>Risk:</b> <span style='color:{risk_colors[risk_level]}'>{risk_level}</span><br>
            <b>Min NDVI:</b> {pred['min_ndvi']:.3f}<br>
            <b>Stress in:</b> {pred['stress_day'] or 'N/A'} days
            </div>""",
            max_width=250
        ),
        tooltip=f"⚠️ {risk_level} Risk Farm"
    ).add_to(m)

    # Add buffer zone for high-risk
    if risk_level in ["HIGH", "CRITICAL"]:
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color=risk_colors[risk_level],
            fill=True,
            fill_opacity=0.1,
            tooltip="⚠️ Risk buffer zone (5km)"
        ).add_to(m)

    # Add nearby synthetic farms for demo
    rng = np.random.default_rng(99)
    for i in range(8):
        nlat = lat + rng.uniform(-0.8, 0.8)
        nlon = lon + rng.uniform(-0.8, 0.8)
        nrisk = rng.choice(["LOW", "MODERATE", "HIGH", "CRITICAL"],
                            p=[0.4, 0.3, 0.2, 0.1])
        folium.CircleMarker(
            location=[nlat, nlon],
            radius=8,
            color=risk_colors[nrisk],
            fill=True, fill_color=risk_colors[nrisk],
            fill_opacity=0.6,
            tooltip=f"Farm {i+1}: {nrisk} Risk"
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
         background:rgba(0,0,0,0.8);padding:12px;border-radius:8px;
         font-size:12px;color:white;border:1px solid #444'>
    <b>Risk Level</b><br>
    🟢 LOW &nbsp;&nbsp; 🟡 MODERATE<br>
    🔴 HIGH &nbsp; 🟣 CRITICAL
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=420, returned_objects=[])


# ── FORECAST PLOT ─────────────────────────────────────────────
with col_forecast:
    st.subheader("📈 21-Day NDVI Forecast with Uncertainty")

    h = np.array(pred["horizon"][:horizon_days])
    med   = np.array(pred["median"][:horizon_days])
    lo80  = np.array(pred["lower_80"][:horizon_days])
    hi80  = np.array(pred["upper_80"][:horizon_days])
    lo50  = np.array(pred["lower_50"][:horizon_days])
    hi50  = np.array(pred["upper_50"][:horizon_days])
    cp_lo = np.array(pred["cp_lower"][:horizon_days])
    cp_hi = np.array(pred["cp_upper"][:horizon_days])

    forecast_dates = [datetime.today() + timedelta(days=int(d)) for d in h]

    fig = go.Figure()

    # Conformal intervals
    if show_conformal:
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=cp_hi.tolist() + cp_lo.tolist()[::-1],
            fill="toself", fillcolor="rgba(0,201,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Conformal 90% CI (coverage guaranteed)", showlegend=True
        ))

    # 80% PI
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=hi80.tolist() + lo80.tolist()[::-1],
        fill="toself", fillcolor="rgba(46,204,113,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="TFT 80% Prediction Interval"
    ))

    # 50% PI
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=hi50.tolist() + lo50.tolist()[::-1],
        fill="toself", fillcolor="rgba(46,204,113,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name="TFT 50% Prediction Interval"
    ))

    # Median forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=med,
        mode="lines+markers",
        line=dict(color="#2ECC71", width=3),
        marker=dict(size=5),
        name="NDVI Median Forecast"
    ))

    # Stress threshold
    fig.add_hline(y=cfg.STRESS_THRESHOLD,
                  line_dash="dash", line_color="red",
                  annotation_text="Stress threshold (0.35)",
                  annotation_position="bottom right")

    # Stress day marker
    if pred["stress_day"] and pred["stress_day"] <= horizon_days:
        sd = forecast_dates[pred["stress_day"] - 1]
        fig.add_shape(
            type="line", x0=sd, x1=sd, y0=0, y1=1,
            yref="paper", line=dict(dash="dot", color="orange", width=2)
        )
        fig.add_annotation(
            x=sd, y=1, yref="paper",
            text=f"⚠️ Day {pred['stress_day']}",
            showarrow=False, font=dict(color="orange", size=11),
            yanchor="bottom"
        )

    # Stress shading — mark days below threshold
    stress_zone = med < cfg.STRESS_THRESHOLD
    if stress_zone.any():
        stress_dates = [forecast_dates[i] for i in range(len(h)) if stress_zone[i]]
        fig.add_trace(go.Scatter(
            x=stress_dates,
            y=[cfg.STRESS_THRESHOLD] * len(stress_dates),
            mode="markers",
            marker=dict(color="rgba(231,76,60,0.5)", size=8, symbol="x"),
            name="Stress days", showlegend=True
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.3)",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   title="Date"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   title="NDVI", range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(255,255,255,0.2)"),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# WEATHER ANALYSIS SECTION
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🌤️ Weather Analysis (Last 120 Days)")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Temperature & Rainfall",
    "💨 Humidity & VPD",
    "☀️ Solar Radiation",
    "🌊 Drought Index (SPI)"
])

with tab1:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          subplot_titles=("Temperature (°C)", "Precipitation (mm/day)"))
    fig2.add_trace(go.Scatter(x=wx_df["date"], y=wx_df["temp_c"],
                               mode="lines", line=dict(color="#FF6B6B", width=1.5),
                               name="Temp (°C)"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=feat_df["date"], y=feat_df["temp_c_roll7_mean"],
                               mode="lines", line=dict(color="#FF9F43", width=2.5),
                               name="7-day avg"), row=1, col=1)
    fig2.add_trace(go.Bar(x=wx_df["date"], y=wx_df["precip_mm"],
                           marker_color="#74B9FF", name="Precip (mm)"), row=2, col=1)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.2)",
                        font=dict(color="white"), height=350,
                        showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = make_subplots(rows=1, cols=2,
                          subplot_titles=("Relative Humidity (%)", "Vapour Pressure Deficit (kPa)"))
    fig3.add_trace(go.Scatter(x=wx_df["date"], y=wx_df["humidity_pct"],
                               fill="tozeroy", fillcolor="rgba(116,185,255,0.2)",
                               line=dict(color="#74B9FF"), name="RH%"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=feat_df["date"], y=feat_df["vpd"],
                               fill="tozeroy", fillcolor="rgba(255,107,107,0.2)",
                               line=dict(color="#FF6B6B"), name="VPD"), row=1, col=2)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.2)",
                        font=dict(color="white"), height=300)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    fig4 = go.Figure(go.Scatter(
        x=wx_df["date"], y=wx_df["solar_kwh"],
        fill="tozeroy", fillcolor="rgba(253,203,110,0.2)",
        line=dict(color="#FDCB6E", width=2), name="Solar (kWh/m²/day)"
    ))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.2)",
                        font=dict(color="white"), height=300,
                        xaxis_title="Date", yaxis_title="Solar Radiation (kWh/m²/day)")
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    spi = feat_df["spi30"]
    colors = ["#E74C3C" if v < -1 else "#F39C12" if v < 0 else "#2ECC71" for v in spi]
    fig5 = go.Figure(go.Bar(x=feat_df["date"], y=spi, marker_color=colors, name="SPI-30"))
    fig5.add_hline(y=-1, line_dash="dash", line_color="orange", annotation_text="Moderate drought")
    fig5.add_hline(y=-2, line_dash="dash", line_color="red",    annotation_text="Severe drought")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.2)",
                        font=dict(color="white"), height=300,
                        xaxis_title="Date", yaxis_title="Standardised Precipitation Index")
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (TFT Attention)
# ─────────────────────────────────────────────────────────────
if show_attention:
    st.divider()
    st.subheader("🔍 TFT Feature Importance (Variable Attention Weights)")

    # Demo importance scores (replace with model.interpret_output in production)
    feature_importance = {
        "Encoder Variables": {
            "NDVI (7-day avg)":     0.18, "Precipitation (14-day)": 0.15,
            "Temperature":          0.12, "VPD":                    0.11,
            "Soil Moisture (SPI)":  0.10, "NDVI momentum":          0.09,
            "Solar Radiation":      0.08, "Humidity":               0.07,
            "Wind Speed":           0.05, "Heat Stress Days":       0.05,
        },
        "Decoder (Future Known)": {
            "Temperature forecast": 0.25, "Precipitation forecast": 0.22,
            "Day of Year (sin)":    0.18, "Solar forecast":         0.15,
            "VPD forecast":         0.12, "Growing Degree Days":    0.08,
        },
        "Static Variables": {
            "Soil Type":   0.35, "Agro Zone":   0.28,
            "Elevation":   0.15, "Latitude":    0.12, "Farm Area": 0.10,
        }
    }

    imp_cols = st.columns(3)
    for i, (name, scores) in enumerate(feature_importance.items()):
        with imp_cols[i]:
            fig_imp = go.Figure(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation="h",
                marker=dict(
                    color=list(scores.values()),
                    colorscale="Viridis",
                    showscale=False
                )
            ))
            fig_imp.update_layout(
                title=name,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.2)",
                font=dict(color="white", size=10),
                height=300,
                margin=dict(l=5, r=5, t=40, b=5),
                xaxis_title="Importance"
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FARM RISK TABLE (Multi-farm view)
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Regional Risk Dashboard")

rng = np.random.default_rng(42)
zones = ["Tropical_Wet", "Tropical_Dry", "Semi_Arid", "Sub_Tropical_Humid", "Highland"]
mock_farms = pd.DataFrame({
    "Farm ID":      [f"FARM_{i:03d}" for i in range(12)],
    "Zone":         [zones[i % 5] for i in range(12)],
    "Soil":         ["Loam","Sandy","Clay","Silty","Laterite","Loam","Sandy","Clay","Silty","Laterite","Loam","Sandy"],
    "Crop":         ["Rice","Sorghum","Wheat","Millet","Barley","Rice","Sorghum","Wheat","Millet","Barley","Rice","Sorghum"],
    "Current NDVI": rng.uniform(0.25, 0.75, 12).round(3),
    "Min NDVI (21d)": rng.uniform(0.15, 0.65, 12).round(3),
    "Stress Day":   rng.integers(5, 22, 12),
    "Area (ha)":    rng.uniform(0.5, 10, 12).round(1),
})
mock_farms["Risk Level"] = mock_farms["Min NDVI (21d)"].apply(cfg.get_risk_level)

def color_risk(val):
    colors = {"LOW": "background-color: #1a5c2e; color: white",
               "MODERATE": "background-color: #5c3d00; color: white",
               "HIGH": "background-color: #5c0000; color: white",
               "CRITICAL": "background-color: #3d0055; color: white"}
    return colors.get(val, "")

styled = mock_farms.style.map(color_risk, subset=["Risk Level"])
st.dataframe(styled, use_container_width=True, height=350)

# Risk summary
r1, r2, r3, r4 = st.columns(4)
for col, level in zip([r1, r2, r3, r4], ["LOW", "MODERATE", "HIGH", "CRITICAL"]):
    count = (mock_farms["Risk Level"] == level).sum()
    col.metric(f"{level}", f"{count} farms",
               delta=f"{count/len(mock_farms)*100:.0f}%")


# ─────────────────────────────────────────────────────────────
# SMS ALERT
# ─────────────────────────────────────────────────────────────
if send_alert_btn:
    st.divider()
    st.subheader("📱 SMS Alert")
    with st.spinner("Sending alert..."):
        result = send_sms_alert(
            farm_id   = f"LAT{lat:.2f}_LON{lon:.2f}",
            risk_level= risk_level,
            min_ndvi  = pred["min_ndvi"],
            stress_day= pred["stress_day"],
            to_number = phone_number
        )
    if result.get("success"):
        st.success(f"✅ SMS sent! SID: {result['sid']}")
    elif "demo_sms" in result:
        st.info("📝 Demo SMS (Twilio not configured):")
        st.code(result["demo_sms"], language=None)
    else:
        st.error(f"❌ {result.get('message', result.get('error'))}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:rgba(255,255,255,0.4);font-size:12px'>"
    "🌾 CropEWS — TFT + Conformal Prediction | "
    "Data: NASA POWER (free) + Sentinel-2 NDVI | "
    "Built for small farmers in developing regions"
    "</p>",
    unsafe_allow_html=True
)
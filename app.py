import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="RoadRisk AI - Authority Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .badge-urgent { background-color: #d9534f; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-priority { background-color: #f0ad4e; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-monitor { background-color: #0275d8; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-normal { background-color: #5cb85c; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADERS ----------------
@st.cache_data
def load_metadata():
    if os.path.exists("data/processed/locations.csv") and os.path.exists("data/processed/road_risk_features.csv"):
        locations = pd.read_csv("data/processed/locations.csv")
        features = pd.read_csv("data/processed/road_risk_features.csv")
        return pd.merge(locations, features, on="location_id")
    else:
        st.error("Data files missing! Please ensure locations.csv and road_risk_features.csv are in data/processed/")
        return pd.DataFrame()

# ---------------- WEATHER SERVICE (Open-Meteo) ----------------
@st.cache_data(ttl=600)
def get_current_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,visibility"
        response = requests.get(url, timeout=3).json()
        current = response.get("current", {})
        return {
            "rainfall": current.get("rain", 0.0),
            "visibility": current.get("visibility", 10000) / 1000.0, # in km
            "temperature": current.get("temperature_2m", 28.0),
            "wind_speed": current.get("wind_speed_10m", 10.0),
            "weather_code": current.get("weather_code", 0)
        }
    except Exception:
        # Offline / simulation fallback values
        return {"rainfall": 18.5, "visibility": 2.4, "temperature": 26.0, "wind_speed": 16.5, "weather_code": 61}

# ---------------- RISK ENGINE ----------------
def compute_risk(row, weather):
    # Standard formula
    hist_comp = row["historical_risk"] * 0.30
    rain_risk = min(weather["rainfall"] / 50.0, 1.0) * 100 * 0.20
    vis_risk = max(0.0, (10.0 - weather["visibility"]) / 10.0) * 100 * 0.15
    traffic_risk = row["traffic_density"] * 100 * 0.15
    damage_risk = row["road_damage_score"] * 100 * 0.15
    water_risk = (100 if row["waterlogging"] == 1 else 0) * 0.05
    
    total_score = int(np.clip(round(hist_comp + rain_risk + vis_risk + traffic_risk + damage_risk + water_risk), 0, 100))
    
    # Category Assignment
    if total_score >= 81:
        category, badge_class = "CRITICAL", "badge-urgent"
    elif total_score >= 61:
        category, badge_class = "HIGH", "badge-priority"
    elif total_score >= 31:
        category, badge_class = "MODERATE", "badge-monitor"
    else:
        category, badge_class = "LOW", "badge-normal"
        
    escalation = total_score - int(row["historical_risk"])
    emerging = escalation >= 15
    
    contributions = [
        {"Factor": "Historical Vulnerability", "Impact": round(hist_comp, 1)},
        {"Factor": "Rainfall & Weather", "Impact": round(rain_risk, 1)},
        {"Factor": "Road Damage & Potholes", "Impact": round(damage_risk, 1)},
        {"Factor": "Traffic Density", "Impact": round(traffic_risk, 1)},
        {"Factor": "Visibility Degradation", "Impact": round(vis_risk, 1)},
        {"Factor": "Waterlogging Presence", "Impact": round(water_risk, 1)}
    ]
    
    return total_score, category, badge_class, escalation, emerging, contributions

# ---------------- RESOURCE PRIORITIZATION LOGIC ----------------
def get_resource_recommendations(row, weather, total_score):
    recommendations = []
    if row["waterlogging"] == 1 or weather["rainfall"] > 15:
        recommendations.append({"Priority": "1 (Urgent)", "Resource": "Drainage & Dewatering Unit", "Reason": "Active waterlogging & surface runoff accumulation"})
    if row["road_damage_score"] > 0.6 or row["pothole_count"] > 5:
        recommendations.append({"Priority": "2 (High)", "Resource": "Rapid Pothole & Road Repair Team", "Reason": f"Severe asphalt degradation ({int(row['pothole_count'])} critical potholes)"})
    if row["traffic_density"] > 0.7:
        recommendations.append({"Priority": "3 (Medium)", "Resource": "Traffic Management & Barricade Squad", "Reason": "Heavy congestion near hazard points"})
    if total_score >= 80 and not recommendations:
        recommendations.append({"Priority": "1 (Urgent)", "Resource": "Emergency Inspection Squad", "Reason": "Severe composite risk score threshold breached"})
    
    if not recommendations:
        recommendations.append({"Priority": "Standard", "Resource": "Routine Monitoring Patrol", "Reason": "Conditions stable within nominal baseline"})
        
    return pd.DataFrame(recommendations)

# ---------------- MAIN APP LAYOUT ----------------
df_locations = load_metadata()

if not df_locations.empty:
    # Sidebar: Monitored Corridors
    st.sidebar.title("🛡️ RoadRisk AI")
    st.sidebar.caption("Authority Decision-Support & Resource Command")
    st.sidebar.markdown("---")

    selected_loc_id = st.sidebar.selectbox(
        "Select Monitored Location:",
        options=df_locations["location_id"].tolist(),
        format_func=lambda x: f"{x} - {df_locations[df_locations['location_id']==x]['name'].values[0]}"
    )

    selected_row = df_locations[df_locations["location_id"] == selected_loc_id].iloc[0]
    weather_data = get_current_weather(selected_row["latitude"], selected_row["longitude"])
    risk_score, risk_cat, badge_class, escalation, is_emerging, drivers = compute_risk(selected_row, weather_data)

    st.sidebar.markdown("---")
    st.sidebar.write("### System Summary")
    st.sidebar.metric("Monitored Zones", len(df_locations))
    st.sidebar.metric("Target Scope", "Internal SIH Round 2026")
    st.sidebar.info("Prototype Mode: Video Computer Vision Cached")

    # ---------------- HEADER SECTION ----------------
    st.title("🚦 Road-Risk Intelligence & Resource Prioritization")
    st.caption(f"Authority Dashboard | Inspecting: **{selected_row['name']}** ({selected_row['corridor_name']})")

    # ---------------- TOP METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Composite Risk Score",
            value=f"{risk_score} / 100",
            delta=f"{'+' if escalation >= 0 else ''}{escalation} vs Historical",
            delta_color="inverse"
        )

    with col2:
        st.markdown(f"**Authority Category**")
        st.markdown(f"### <span class='{badge_class}'>{risk_cat}</span>", unsafe_allow_html=True)

    with col3:
        st.metric(
            label="Live Rainfall (Open-Meteo)",
            value=f"{weather_data['rainfall']} mm/h",
            delta=f"Vis: {weather_data['visibility']} km"
        )

    with col4:
        st.metric(
            label="Vision Detections",
            value=f"{int(selected_row['pothole_count'])} Potholes",
            delta="Waterlogged" if selected_row['waterlogging'] == 1 else "Normal Surface",
            delta_color="inverse" if selected_row['waterlogging'] == 1 else "normal"
        )

    if is_emerging:
        st.error(f"⚠️ **EMERGING RISK DETECTED:** Current conditions are escalating rapidly (+{escalation} points) compared to historical baseline. Urgent resource allocation recommended.")

    st.markdown("---")

    # ---------------- MIDDLE ROW: VIDEO & EXPLAINABILITY ----------------
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.subheader("📹 Recorded / Simulated Road-Camera Feed")
        video_path = selected_row["video_file"]
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning(f"Video file not found at `{video_path}`. Place your mp4 files in the `videos/` folder.")
            st.info("Showing simulated camera metrics: Traffic Density: {:.0%}, Road Damage: {:.0%}".format(
                selected_row["traffic_density"], selected_row["road_damage_score"]
            ))
        
        st.caption("Feed Source: Fixed Infrastructure Camera | Analysis Pipeline: YOLOv8 (Cached Extraction)")

    with right_col:
        st.subheader("📊 Explainable AI (SHAP Factor Breakdown)")
        st.write("Why is this location receiving this risk score?")
        
        chart_df = pd.DataFrame(drivers).sort_values("Impact", ascending=True)
        chart = alt.Chart(chart_df).mark_bar(color='#ff4b4b').encode(
            x=alt.X('Impact:Q', title='Contribution to Risk Score (Points)'),
            y=alt.Y('Factor:N', sort='-x', title=''),
            tooltip=['Factor', 'Impact']
        ).properties(height=260)
        
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ---------------- BOTTOM ROW: RESOURCES & AUTHORITY BRIEF ----------------
    b_col1, b_col2 = st.columns([1.2, 1])

    with b_col1:
        st.subheader("🛠️ Recommended Resource Prioritization")
        rec_df = get_resource_recommendations(selected_row, weather_data, risk_score)
        st.table(rec_df)

    with b_col2:
        st.subheader("📝 GenAI Authority Brief")
        top_driver = max(drivers, key=lambda x: x['Impact'])
        brief_text = (
            f"**{selected_row['name']}** is currently flagged as **{risk_cat}** urgency (Score: {risk_score}/100). "
            f"The primary driver for escalation is **{top_driver['Factor']}** contributing +{top_driver['Impact']} points. "
            f"{'Immediate drainage de-watering squads and asphalt cold-patch teams must be dispatched' if risk_score > 60 else 'Standard surveillance routine is adequate for this period.'}"
        )
        st.info(brief_text)
        
        with st.expander("Inspect Raw Multi-Signal Feature Vector"):
            st.json({
                "location_id": selected_row["location_id"],
                "historical_baseline": int(selected_row["historical_risk"]),
                "historical_accidents": int(selected_row["historical_accidents"]),
                "live_weather": weather_data,
                "vision_features": {
                    "vehicle_count": int(selected_row["vehicle_count"]),
                    "traffic_density": float(selected_row["traffic_density"]),
                    "potholes_detected": int(selected_row["pothole_count"]),
                    "waterlogging": bool(selected_row["waterlogging"])
                },
                "output_risk_score": risk_score
            })
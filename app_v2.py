import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import os
import folium
from streamlit_folium import st_folium

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="RoadRisk AI - Command Center V2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card { background-color: #1a1c24; border-radius: 8px; padding: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
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
    return pd.DataFrame()

# ---------------- WEATHER SERVICE ----------------
@st.cache_data(ttl=600)
def get_current_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,visibility"
        response = requests.get(url, timeout=3).json()
        current = response.get("current", {})
        return {
            "rainfall": current.get("rain", 0.0),
            "visibility": current.get("visibility", 10000) / 1000.0,
            "temperature": current.get("temperature_2m", 28.0),
            "wind_speed": current.get("wind_speed_10m", 10.0),
            "weather_code": current.get("weather_code", 0)
        }
    except Exception:
        return {"rainfall": 18.5, "visibility": 2.4, "temperature": 26.0, "wind_speed": 16.5, "weather_code": 61}

# ---------------- RISK ENGINE ----------------
def compute_risk(row, weather):
    hist_comp = row["historical_risk"] * 0.30
    rain_risk = min(weather["rainfall"] / 50.0, 1.0) * 100 * 0.20
    vis_risk = max(0.0, (10.0 - weather["visibility"]) / 10.0) * 100 * 0.15
    traffic_risk = row["traffic_density"] * 100 * 0.15
    damage_risk = row["road_damage_score"] * 100 * 0.15
    water_risk = (100 if row["waterlogging"] == 1 else 0) * 0.05
    
    total_score = int(np.clip(round(hist_comp + rain_risk + vis_risk + traffic_risk + damage_risk + water_risk), 0, 100))
    
    if total_score >= 81: category, badge_class = "CRITICAL", "badge-urgent"
    elif total_score >= 61: category, badge_class = "HIGH", "badge-priority"
    elif total_score >= 31: category, badge_class = "MODERATE", "badge-monitor"
    else: category, badge_class = "LOW", "badge-normal"
        
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

def get_resource_recommendations(row, weather, total_score):
    recommendations = []
    if row["waterlogging"] == 1 or weather["rainfall"] > 15:
        recommendations.append({"Priority": "1 (Urgent)", "Resource": "Drainage Unit", "Reason": "Active waterlogging"})
    if row["road_damage_score"] > 0.6 or row["pothole_count"] > 5:
        recommendations.append({"Priority": "2 (High)", "Resource": "Pothole Repair Team", "Reason": "Severe asphalt degradation"})
    if row["traffic_density"] > 0.7:
        recommendations.append({"Priority": "3 (Medium)", "Resource": "Traffic Squad", "Reason": "Heavy congestion"})
    if total_score >= 80 and not recommendations:
        recommendations.append({"Priority": "1 (Urgent)", "Resource": "Emergency Squad", "Reason": "Risk threshold breached"})
    if not recommendations:
        recommendations.append({"Priority": "Standard", "Resource": "Monitoring Patrol", "Reason": "Conditions stable"})
    return pd.DataFrame(recommendations)

# ---------------- MAIN APP LAYOUT ----------------
df_locations = load_metadata()

if not df_locations.empty:
    st.sidebar.title("🛡️ RoadRisk AI (V2)")
    selected_loc_id = st.sidebar.selectbox("Select Monitored Location:", options=df_locations["location_id"].tolist(), format_func=lambda x: f"{x} - {df_locations[df_locations['location_id']==x]['name'].values[0]}")
    
    selected_row = df_locations[df_locations["location_id"] == selected_loc_id].iloc[0]
    weather_data = get_current_weather(selected_row["latitude"], selected_row["longitude"])
    risk_score, risk_cat, badge_class, escalation, is_emerging, drivers = compute_risk(selected_row, weather_data)

    st.title("🚦 Road-Risk Intelligence & Resource Prioritization")
    st.caption(f"Authority Dashboard | Inspecting: **{selected_row['name']}**")

    # ---------------- METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Composite Risk Score", f"{risk_score}/100", f"{'+' if escalation >= 0 else ''}{escalation} vs Hist", delta_color="inverse")
    col2.markdown(f"**Authority Category**<br>### <span class='{badge_class}'>{risk_cat}</span>", unsafe_allow_html=True)
    col3.metric("Live Rainfall", f"{weather_data['rainfall']} mm/h", f"Vis: {weather_data['visibility']} km")
    col4.metric("Vision Detections", f"{int(selected_row['pothole_count'])} Potholes", "Waterlogged" if selected_row['waterlogging'] == 1 else "Normal Surface", delta_color="inverse" if selected_row['waterlogging'] == 1 else "normal")

    if is_emerging: st.error(f"⚠️ **EMERGING RISK DETECTED:** Current conditions are escalating rapidly (+{escalation} points).")
    st.markdown("---")

    # ---------------- INTERACTIVE MAP & CHART ----------------
    map_col, chart_col = st.columns(2)
    
    with map_col:
        st.subheader("🗺️ Live Geographic Risk Map")
        # Create interactive map
        m = folium.Map(location=[selected_row['latitude'], selected_row['longitude']], zoom_start=11)
        for _, r in df_locations.iterrows():
            marker_color = 'red' if r['location_id'] == 'L4' else 'orange' if r['location_id'] == 'L3' else 'blue' if r['location_id'] == 'L2' else 'green'
            folium.Marker([r['latitude'], r['longitude']], popup=r['name'], icon=folium.Icon(color=marker_color, icon='info-sign')).add_to(m)
        st_folium(m, width=500, height=300)

    with chart_col:
        st.subheader("📈 24-Hour Risk Escalation Trend")
        # Simulate trend data for the graph
        hours = [f"{i}:00" for i in range(1, 25)]
        trend_scores = [max(0, risk_score - np.random.randint(0, 20)) for _ in range(23)] + [risk_score]
        trend_df = pd.DataFrame({"Time": hours, "Risk Score": trend_scores})
        
        trend_chart = alt.Chart(trend_df).mark_area(opacity=0.4, color='#ff4b4b').encode(
            x=alt.X('Time', sort=None),
            y=alt.Y('Risk Score', scale=alt.Scale(domain=[0, 100]))
        ) + alt.Chart(trend_df).mark_line(color='#ff4b4b').encode(
            x=alt.X('Time', sort=None),
            y='Risk Score'
        )
        st.altair_chart(trend_chart, use_container_width=True)

    st.markdown("---")

    # ---------------- BOTTOM ROW ----------------
    b_col1, b_col2 = st.columns([1, 1.2])

    with b_col1:
        st.subheader("📹 Road-Camera Feed")
        video_path = selected_row["video_file"]
        if os.path.exists(video_path): st.video(video_path)
        else: st.warning(f"Simulated feed. Traffic Density: {selected_row['traffic_density']:.0%}")

    with b_col2:
        st.subheader("🛠️ Recommended Resources & Brief")
        st.table(get_resource_recommendations(selected_row, weather_data, risk_score))
        
        # Export Report Button
        csv_data = pd.DataFrame(drivers).to_csv(index=False)
        st.download_button(
            label="📥 Download Official Authority Risk Report (CSV)",
            data=csv_data,
            file_name=f"Risk_Report_{selected_loc_id}.csv",
            mime="text/csv"
        )
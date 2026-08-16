import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="RoadRisk AI - Pro Edition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Sleek UI Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Helvetica Neue', sans-serif; }
    .risk-critical { color: #ff4b4b; font-size: 2.5rem; font-weight: 900; text-shadow: 0px 0px 10px rgba(255,75,75,0.5); }
    .risk-high { color: #f0ad4e; font-size: 2.5rem; font-weight: 900; }
    .risk-moderate { color: #0275d8; font-size: 2.5rem; font-weight: 900; }
    .risk-low { color: #5cb85c; font-size: 2.5rem; font-weight: 900; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 60px; font-size: 1.2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATA & LOGIC ----------------
@st.cache_data
def load_metadata():
    if os.path.exists("data/processed/locations.csv") and os.path.exists("data/processed/road_risk_features.csv"):
        locations = pd.read_csv("data/processed/locations.csv")
        features = pd.read_csv("data/processed/road_risk_features.csv")
        return pd.merge(locations, features, on="location_id")
    return pd.DataFrame()

def compute_risk(hist_risk, rainfall, visibility, traffic_density, road_damage, waterlogging):
    hist_comp = hist_risk * 0.30
    rain_risk = min(rainfall / 50.0, 1.0) * 100 * 0.20
    vis_risk = max(0.0, (10.0 - visibility) / 10.0) * 100 * 0.15
    traffic_comp = traffic_density * 100 * 0.15
    damage_comp = road_damage * 100 * 0.15
    water_comp = (100 if waterlogging else 0) * 0.05
    
    total_score = int(np.clip(round(hist_comp + rain_risk + vis_risk + traffic_comp + damage_comp + water_comp), 0, 100))
    
    if total_score >= 81: return total_score, "CRITICAL", "risk-critical", "videos/location_4.mp4"
    elif total_score >= 61: return total_score, "HIGH", "risk-high", "videos/location_3.mp4"
    elif total_score >= 31: return total_score, "MODERATE", "risk-moderate", "videos/location_2.mp4"
    else: return total_score, "LOW", "risk-low", "videos/location_1.mp4"

# ---------------- MAIN APP ----------------
df_locations = load_metadata()

if not df_locations.empty:
    st.title("⚡ RoadRisk AI: Predictive Command Center")
    st.markdown("Dynamic Authority Decision-Support & Resource Allocation Platform")
    
    # Location Selector at the top instead of sidebar for a cleaner look
    selected_loc_id = st.selectbox("📍 Select Regional Corridor:", options=df_locations["location_id"].tolist(), format_func=lambda x: f"{df_locations[df_locations['location_id']==x]['name'].values[0]} ({x})")
    row = df_locations[df_locations["location_id"] == selected_loc_id].iloc[0]
    
    # Create beautiful tabs
    tab1, tab2 = st.tabs(["🔴 Live Risk Dashboard", "🎛️ What-If Risk Simulator"])
    
    # --- TAB 1: LIVE DASHBOARD ---
    with tab1:
        st.markdown("### Current Live Intelligence")
        # Base values for live view
        live_rain = 18.5 if row['location_id'] in ['L3','L4'] else 0.0
        live_vis = 2.4 if row['location_id'] in ['L3','L4'] else 10.0
        
        l_score, l_cat, l_css, l_video = compute_risk(row['historical_risk'], live_rain, live_vis, row['traffic_density'], row['road_damage_score'], bool(row['waterlogging']))
        
        col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
        with col_m1:
            st.markdown("### Risk Score")
            st.markdown(f"<div class='{l_css}'>{l_score}%</div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown("### Status")
            st.markdown(f"<div class='{l_css}'>{l_cat}</div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown("### Live Camera Feed")
            if os.path.exists(l_video): st.video(l_video)
            else: st.warning(f"Video {l_video} missing.")

    # --- TAB 2: WHAT-IF SIMULATOR ---
    with tab2:
        st.markdown("### 🛠️ Intervention Effectiveness Simulator")
        st.write("Adjust current environmental and traffic conditions to see how specific interventions (like clearing traffic or pumping water) reduce the overall risk score.")
        
        s_col1, s_col2 = st.columns([1.5, 1])
        
        with s_col1:
            st.markdown("#### Adjust Parameters")
            sim_rain = st.slider("🌧️ Rainfall (mm/h) - Weather condition", 0.0, 100.0, float(live_rain))
            sim_traffic = st.slider("🚗 Traffic Density (0 = Empty, 1 = Gridlock)", 0.0, 1.0, float(row['traffic_density']))
            sim_damage = st.slider("🛣️ Road Damage Severity", 0.0, 1.0, float(row['road_damage_score']))
            sim_water = st.toggle("🌊 Standing Water (Waterlogging present?)", value=bool(row['waterlogging']))
            
        with s_col2:
            # Recalculate based on sliders
            s_score, s_cat, s_css, s_video = compute_risk(row['historical_risk'], sim_rain, live_vis, sim_traffic, sim_damage, sim_water)
            
            st.markdown("#### Projected Risk Level")
            st.markdown(f"<div class='{s_css}'>{s_score}%  ({s_cat})</div>", unsafe_allow_html=True)
            
            diff = l_score - s_score
            if diff > 0:
                st.success(f"✅ Interventions reduced risk by **{diff} points**.")
            elif diff < 0:
                st.error(f"⚠️ Conditions worsened risk by **{abs(diff)} points**.")
            else:
                st.info("➖ Risk remains unchanged.")
                
            st.markdown("#### Projected Road Condition")
            if os.path.exists(s_video): st.video(s_video)
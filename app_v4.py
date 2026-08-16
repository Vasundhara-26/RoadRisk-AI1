import streamlit as st
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="RoadRisk AI - Vision Edition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
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

# Determine base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

def find_video_path(filename_stem):
    """Finds video path matching filename or variations (e.g. location_1 or location_1.mp4.mp4)"""
    possible_names = [
        f"{filename_stem}.mp4",
        f"{filename_stem}.mp4.mp4",
        f"{filename_stem}.MP4",
        filename_stem
    ]
    # Check in videos folder
    for name in possible_names:
        p = os.path.join(VIDEOS_DIR, name)
        if os.path.exists(p):
            return p
    # Check in root directory
    for name in possible_names:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    return None

# ---------------- AI & DATA LOGIC ----------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

def compute_risk(hist_risk, rainfall, visibility, traffic_density, road_damage, waterlogging):
    hist_comp = hist_risk * 0.30
    rain_risk = min(rainfall / 50.0, 1.0) * 100 * 0.20
    vis_risk = max(0.0, (10.0 - visibility) / 10.0) * 100 * 0.15
    traffic_comp = traffic_density * 100 * 0.15
    damage_comp = road_damage * 100 * 0.15
    water_comp = (100 if waterlogging else 0) * 0.05
    
    total_score = int(np.clip(round(hist_comp + rain_risk + vis_risk + traffic_comp + damage_comp + water_comp), 0, 100))
    
    if total_score >= 81: return total_score, "CRITICAL", "risk-critical", "location_4"
    elif total_score >= 61: return total_score, "HIGH", "risk-high", "location_3"
    elif total_score >= 31: return total_score, "MODERATE", "risk-moderate", "location_2"
    else: return total_score, "LOW", "risk-low", "location_1"

def get_recommendations(risk_category):
    if risk_category == "CRITICAL":
        return [
            "Immediately dispatch emergency response teams.",
            "Close the road/corridor to civilian traffic.",
            "Broadcast emergency alerts to the public.",
            "Mobilize heavy equipment for hazard clearance."
        ]
    elif risk_category == "HIGH":
        return [
            "Deploy traffic management personnel.",
            "Reduce speed limits dynamically via VMS.",
            "Monitor situation continuously with drones/cameras.",
            "Prepare standby maintenance crews."
        ]
    elif risk_category == "MODERATE":
        return [
            "Issue advisory warnings to drivers.",
            "Increase automated monitoring frequency.",
            "Schedule standard maintenance review."
        ]
    else:
        return [
            "Normal conditions. Continue standard operations.",
            "Maintain routine monitoring."
        ]

# ---------------- MAIN APP ----------------
st.title("RoadRisk AI: Autonomous Vision Command")
st.markdown("Dynamic Authority Decision-Support & Real-Time Computer Vision Platform")

loc_data = {
    "L1": {"name": "Sector 9 Intersection", "hist": 20, "traffic": 0.3, "damage": 0.1, "water": False, "rain": 0.0, "vis": 10.0},
    "L2": {"name": "Industrial Bypass", "hist": 45, "traffic": 0.6, "damage": 0.8, "water": False, "rain": 0.0, "vis": 8.0},
    "L3": {"name": "Highway 44 Corridor", "hist": 70, "traffic": 0.8, "damage": 0.9, "water": False, "rain": 25.0, "vis": 4.0},
    "L4": {"name": "Low-Lying Urban Sector", "hist": 85, "traffic": 0.9, "damage": 0.5, "water": True, "rain": 45.0, "vis": 2.0}
}

selected_loc_id = st.selectbox("Select Monitored Corridor:", options=list(loc_data.keys()), format_func=lambda x: f"{loc_data[x]['name']} ({x})")
row = loc_data[selected_loc_id]

# --- LIVE VISION DASHBOARD ---
l_score, l_cat, l_css, l_stem = compute_risk(row['hist'], row['rain'], row['vis'], row['traffic'], row['damage'], row['water'])
video_actual_path = find_video_path(l_stem)

col_video, col_score, col_status = st.columns([2, 1, 1])

with col_video:
    st.markdown("### Live AI Camera Feed")
    stframe = st.empty()
    
with col_score:
    st.markdown("### Risk Score")
    score_placeholder = st.empty()
    
with col_status:
    st.markdown("### Status")
    status_placeholder = st.empty()
    
if video_actual_path and os.path.exists(video_actual_path):
    score_placeholder.markdown("<div style='font-size: 2.5rem; color: gray; font-weight: bold;'>Detecting...</div>", unsafe_allow_html=True)
    status_placeholder.markdown("<div style='font-size: 2.5rem; color: gray; font-weight: bold;'>Detecting...</div>", unsafe_allow_html=True)
    
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_actual_path)
    
    start_time = time.time()
    while cap.isOpened():
        if time.time() - start_time > 2.0:
            break
        ret, frame = cap.read()
        if not ret:
            break 
        
        results = model(frame, conf=0.25)
        annotated_frame = results[0].plot()
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        
    cap.release()
    
    score_placeholder.markdown(f"<div class='{l_css}'>{l_score}%</div>", unsafe_allow_html=True)
    status_placeholder.markdown(f"<div class='{l_css}'>{l_cat}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Recommended Authority Actions")
    actions = get_recommendations(l_cat)
    for action in actions:
        st.markdown(f"- **{action}**")
        
else: 
    with col_video:
        st.error(f"Looking for: `{l_stem}.mp4`")
        existing_files = os.listdir(VIDEOS_DIR) if os.path.exists(VIDEOS_DIR) else os.listdir(BASE_DIR)
        st.warning(f"Files found in folder: {existing_files}")
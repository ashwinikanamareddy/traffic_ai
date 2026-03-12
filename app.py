import streamlit as st
import importlib

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="UrbanFlow AI - Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HACKATHON DEMO MODE BANNER ──
st.warning("🚀 UrbanFlow AI Smart City Traffic Control Platform — Hackathon Demonstration Mode (Auth Disabled)")

# ── SESSION STATE INITIALIZATION ──
if "processed" not in st.session_state:
    st.session_state.processed = False
if "df" not in st.session_state:
    st.session_state.df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_vehicles": 0,
        "queue_count": 0,
        "total_violations": 0,
        "red_light_violations": 0,
        "rash_driving": 0,
        "no_helmet_violations": 0,
        "mobile_usage_violations": 0,
        "triple_riding_violations": 0,
        "heavy_load_violations": 0,
        "autos": 0,
    }
if "page" not in st.session_state:
    st.session_state.page = "DashboardOverview"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "show_guided_tour" not in st.session_state:
    st.session_state.show_guided_tour = False
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "vehicle_counts" not in st.session_state:
    st.session_state.vehicle_counts = {
        "cars": 0,
        "bikes": 0,
        "buses": 0,
        "trucks": 0,
        "autos": 0,
    }
if "live_paused" not in st.session_state:
    st.session_state.live_paused = False
if "live_video_path" not in st.session_state:
    st.session_state.live_video_path = None
if "live_frame_index" not in st.session_state:
    st.session_state.live_frame_index = 0
if "live_event_log" not in st.session_state:
    st.session_state.live_event_log = []
if "violations" not in st.session_state:
    st.session_state.violations = []
if "selected_violation" not in st.session_state:
    st.session_state.selected_violation = None

# -- Initialize Control Engine --
try:
    from backend.city_control_engine import get_control_engine
    engine = get_control_engine()
except Exception as e:
    st.error(f"Engine initialization error: {e}")
    st.stop()

# ── HELPER FUNCTIONS ──
def _load_view_module(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        st.error(f"Failed to load module `{module_name}`.")
        st.exception(exc)
        return None

# ── GLOBAL STYLES ──
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F7F9FC !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        min-width: 240px !important;
        max-width: 240px !important;
    }
    
    /* Sidebar Section Header */
    .sidebar-title {
        font-size: 12px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0 16px 12px 16px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 24px;
        margin-top: 12px;
    }

    /* Modern Radio Buttons for clean navigation */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 0 8px 4px 8px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #F8FAFC;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: #EFF6FF;
        border-color: #BFDBFE;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child {
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 600;
        color: #475569;
        font-size: 14px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) > div:last-child {
        color: #2563EB;
    }
    
    /* Hide radio native elements */
    [data-testid="stSidebar"] input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ── GLOBAL TOP NAVBAR ──
with st.container():
    nc1, nc2, nc3 = st.columns([3, 4, 3])
    
    # Left: Branding
    with nc1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-top:8px;">
            <div style="background:#EFF6FF; color:#2563EB; width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; border:1px solid #BFDBFE;">🚦</div>
            <div>
                <div style="font-size:18px; font-weight:800; color:#0F172A; line-height:1.2;">UrbanFlow AI</div>
                <div style="font-size:11px; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">City Traffic Command Center</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Middle: Quick Scenarios
    with nc2:
        st.markdown("<div style='font-size:11px; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:4px; text-align:center;'>Quick Scenario Controls</div>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        if sc1.button("🚑 Ambulance", use_container_width=True):
            engine.process_event("ambulance_detected", {"location": "INT-03"})
            st.rerun()
        if sc2.button("🚗 Heavy Traffic", use_container_width=True):
            engine.process_event("vehicle_detected", {"counts": {"cars": 150, "buses": 10}})
            st.rerun()
        if sc3.button("💥 Accident", use_container_width=True):
            engine.process_event("incident_triggered", {"type": "Major Accident", "location": "INT-02"})
            st.rerun()
        if sc4.button("🔄 Reset", use_container_width=True):
            engine.process_event("emergency_cleared")
            engine.process_event("incident_cleared")
            st.rerun()
            
    # Right: Presentation Tools
    with nc3:
        st.markdown("<div style='font-size:11px; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:4px; text-align:right;'>Presentation Tools</div>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        if pc1.button("🎓 Guided Tour", use_container_width=True):
            st.session_state.show_guided_tour = not st.session_state.show_guided_tour
            st.rerun()
        if pc2.button("📊 Demo Report", use_container_width=True):
            st.toast("Generating Demo Report...", icon="⏳")
            report_txt = (
                f"**UrbanFlow Engine Report**\n\n"
                f"Density: {engine.state['density_level']}\n"
                f"Emergency Mode: {engine.state['active_emergency']}\n"
                f"Incident State: {engine.state['active_incident']}"
            )
            st.info(report_txt)
            
st.markdown("<hr style='margin:12px 0 24px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)

# ── SYSTEM PERFORMANCE METRICS ──
st.markdown("""
<div style="display:flex; justify-content:space-between; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 24px; margin-bottom:24px;">
    <div style="display:flex; align-items:center; gap:8px;"><span style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">Vision FPS:</span> <span style="color:#0F172A; font-weight:800; font-size:14px;">30.0</span></div>
    <div style="display:flex; align-items:center; gap:8px;"><span style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">Detection Latency:</span> <span style="color:#0F172A; font-weight:800; font-size:14px;">42ms</span></div>
    <div style="display:flex; align-items:center; gap:8px;"><span style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">AI Brain Decision Time:</span> <span style="color:#22C55E; font-weight:800; font-size:14px;">12ms</span></div>
    <div style="display:flex; align-items:center; gap:8px;"><span style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">Signal Network Sync:</span> <span style="color:#2563EB; font-weight:800; font-size:14px;">< 1ms</span></div>
</div>
""", unsafe_allow_html=True)

# ── GUIDED TOUR OVERLAY ──
if st.session_state.show_guided_tour:
    st.markdown(r"""
    <div style="background:#0F172A; border-radius:12px; padding:24px; margin-bottom:24px; color:#F8FAFC; box-shadow:0 20px 25px -5px rgba(0,0,0,0.2);">
        <h3 style="margin:0 0 16px 0; font-size:18px; color:#38BDF8;">🎓 UrbanFlow AI: Guided Demo Presentation</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
            <div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 1</span> Upload Camera Feed <i>(Live Monitoring)</i></div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 2</span> AI Detection & Density Tracking</div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 3</span> Dynamic Signal Optimization</div>
            </div>
            <div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 4</span> Trigger Emergency Mode <i>(Ambulance)</i></div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 5</span> Watch AI Rerouting in Action</div>
                <div style="margin-bottom:12px;"><span style="background:#38BDF8; color:#0F172A; padding:2px 8px; border-radius:4px; font-weight:800; font-size:11px; margin-right:8px;">STEP 6</span> Global Analytics & Heatmaps</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN NAVIGATION SIDEBAR ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">Main Navigation</div>', unsafe_allow_html=True)

    menu_items = [
        ("Dashboard Overview", "DashboardOverview", "📊"),
        ("City Traffic Intelligence", "CityIntelligence", "🧠"),
        ("Live Traffic Monitoring", "LiveMonitoring", "📹"),
        ("Traffic Density Analyzer", "TrafficDensity", "📈"),
        ("Traffic Prediction AI", "TrafficPrediction", "🔮"),
        ("AI Traffic Brain", "AIBrain", "⚙️"),
        ("Emergency Response", "EmergencyResponse", "🚨"),
        ("Route Optimization", "RouteOptimization", "🗺️"),
        ("City Traffic Map", "CityMap", "🏙️"),
        ("Scenario Simulator", "ScenarioSimulator", "🕹️"),
        ("Traffic Incident Manager", "TrafficIncidentManager", "⚠️"),
        ("Traffic Analytics", "TrafficAnalytics", "📊"),
        ("AI Traffic Assistant", "AIAssistant", "💬"),
    ]

    labels = [f"{icon} {label}" for label, key, icon in menu_items]
    keys = [key for _, key, _ in menu_items]
    
    current_page = st.session_state.get("page", "DashboardOverview")
    default_idx = keys.index(current_page) if current_page in keys else 0
    
    selection = st.radio("Navigation", labels, index=default_idx, label_visibility="collapsed")
    sel_index = labels.index(selection)
    st.session_state.page = keys[sel_index]
    
    # Footer Mission Statement
    st.markdown("""
    <div style="margin-top:auto; padding-top:40px; font-size:11px; color:#94A3B8; line-height:1.5;">
        <strong style="color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">UrbanFlow Operations</strong><br>
        A city-wide intelligent traffic system optimizing multi-lane grid movement through automated Green Corridors and AI vision.
    </div>
    """, unsafe_allow_html=True)

# ── PAGE ROUTING ──
page_to_module = {
    "DashboardOverview": "views.dashboard_overview",
    "CityIntelligence": "views.city_intelligence",
    "LiveMonitoring": "views.live_monitoring",
    "TrafficDensity": "views.traffic_density",
    "TrafficPrediction": "views.traffic_prediction",
    "AIBrain": "views.ai_brain",
    "EmergencyResponse": "views.emergency_response",
    "RouteOptimization": "views.route_optimization",
    "CityMap": "views.city_map",
    "ScenarioSimulator": "views.scenario_simulator",
    "TrafficIncidentManager": "views.traffic_incident_manager",
    "TrafficAnalytics": "views.traffic_analytics",
    "AIAssistant": "views.ai_assistant",
}

module_name = page_to_module.get(st.session_state.page, "views.dashboard_overview")

try:
    page_module = _load_view_module(module_name)
    if page_module and hasattr(page_module, "show"):
        page_module.show()
    else:
        st.error(f"Module {module_name} could not be rendered.")
except Exception as e:
    st.markdown(r"""
    <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:12px; padding:20px; display:flex; gap:16px; align-items:center;">
        <div style="font-size:32px;">⚠️</div>
        <div>
            <h3 style="margin:0 0 4px 0; color:#DC2626; font-size:16px;">Module Temporarily Unavailable</h3>
            <div style="color:#991B1B; font-size:14px;">A runtime error occurred in this view. Core traffic monitoring & background Engine logic remains active.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"Stack Trace: {e}")

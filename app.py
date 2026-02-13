import streamlit as st

from backend.auth import init_db
import views.dashboard as dashboard
import views.export as export
import views.insights as insights
import views.live_feed as live_feed
import views.login as login
import views.queue as queue
import views.statistics as statistics
import views.violation_evidence as violation_evidence
import views.violations as violations


st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

if "processed" not in st.session_state:
    st.session_state.processed = False
if "df" not in st.session_state:
    st.session_state.df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_vehicles": 0,
        "queue_count": 0,
        "red_light_violations": 0,
        "rash_driving": 0,
    }
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "vehicle_counts" not in st.session_state:
    st.session_state.vehicle_counts = {
        "cars": 0,
        "bikes": 0,
        "buses": 0,
        "trucks": 0,
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

if not st.session_state.logged_in:
    login.show()
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #eef2f7;
        }
        .sb-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 6px 14px 6px;
        }
        .sb-logo {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #14b8a6, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
        }
        .sb-title {
            font-size: 16px;
            font-weight: 800;
            color: #0f172a;
        }
        .sb-sub {
            font-size: 12px;
            color: #64748b;
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: 8px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: #ffffff;
            border: 1px solid #e6edf4;
            border-radius: 12px;
            padding: 10px 12px;
            margin: 0 0 8px 0;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: #ecfeff;
            border-color: #c6f7f1;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label::before {
            content: "";
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1.5px solid #cbd5f5;
            display: inline-block;
            margin-right: 10px;
            flex-shrink: 0;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::before {
            background: #22c55e;
            border-color: #22c55e;
            box-shadow: 0 0 0 3px rgba(34,197,94,0.15);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label > div {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            color: #0f172a;
            font-size: 14px;
        }
        [data-testid="stSidebar"] input[type="radio"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-logo">TA</div>
            <div>
                <div class="sb-title">TrafficAI</div>
                <div class="sb-sub">Intelligence System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu_items = [
        ("Dashboard Overview", "Dashboard", "▦"),
        ("Live Video Feed", "LiveFeed", "🎥"),
        ("Queue Analytics", "Queue", "🚗"),
        ("Violation Detection", "Violations", "⚠️"),
        ("Violation Evidence", "ViolationEvidence", "📷"),
        ("Vehicle Statistics", "Statistics", "📊"),
        ("Export Reports", "Export", "⬇️"),
        ("Trends & Insights", "Insights", "📈"),
    ]

    labels = [f"{icon}  {label}" for label, _, icon in menu_items]
    keys = [key for _, key, _ in menu_items]
    current_key = st.session_state.get("page", "Dashboard")
    default_index = keys.index(current_key) if current_key in keys else 0

    selection = st.radio("Navigation", labels, index=default_index, label_visibility="collapsed")
    sel_index = labels.index(selection)
    st.session_state.page = keys[sel_index]

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

page = st.session_state.get("page", "Dashboard")

if page == "Dashboard":
    dashboard.show()
elif page == "Upload":
    live_feed.show()
elif page == "LiveFeed":
    live_feed.show()
elif page == "Queue":
    queue.show()
elif page == "Violations":
    violations.show()
elif page == "ViolationEvidence":
    violation_evidence.show()
elif page == "Statistics":
    statistics.show()
elif page == "Export":
    export.show()
elif page == "Insights":
    insights.show()

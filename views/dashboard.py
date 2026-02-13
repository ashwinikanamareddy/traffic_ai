from datetime import datetime
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from backend.process_video import process_full_video


def _init_state():
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
    if "dash_search" not in st.session_state:
        st.session_state.dash_search = ""


def _to_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_df():
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return None


def _save_uploaded_file(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    path = os.path.join("uploads", safe_name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _format_time_from_frame(frame, fps=30):
    seconds = max(0, _to_int(frame, 0) // fps)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def _lucide_svg(icon_name: str) -> str:
    icons = {
        "camera": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z"/>'
            '<circle cx="12" cy="13" r="3"/>'
            "</svg>"
        ),
        "car": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M3 12h18l-1.4-4.2A2 2 0 0 0 17.7 6H6.3a2 2 0 0 0-1.9 1.8z"/>'
            '<path d="M3 12v5h2"/><path d="M21 12v5h-2"/>'
            '<circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/>'
            "</svg>"
        ),
        "alert-triangle": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="m10.29 3.86-7.82 13.54A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.73-2.6L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<path d="M12 9v4"/><path d="M12 17h.01"/>'
            "</svg>"
        ),
        "map-pin": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>'
            '<circle cx="12" cy="10" r="3"/>'
            "</svg>"
        ),
    }
    return icons.get(icon_name, "")


def _avg_speed(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for col in ["speed", "avg_speed", "average_speed"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.mean())
    metrics = st.session_state.get("metrics", {}) or {}
    for key in ["avg_speed", "average_speed"]:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


def _traffic_peak_percent(df: pd.DataFrame):
    if df is None or df.empty or "timestamp" not in df.columns:
        return 0.0
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().all() or "total_vehicles" not in df.columns:
        return 0.0
    frame_df = df.drop_duplicates(subset=["frame"])
    frame_df = frame_df.assign(hour=ts.loc[frame_df.index].dt.hour)
    totals = frame_df.groupby("hour")["total_vehicles"].sum()
    total_sum = float(totals.sum()) if not totals.empty else 0.0
    peak = float(totals.max()) if not totals.empty else 0.0
    return 0.0 if total_sum == 0 else (peak / total_sum) * 100.0


def _congestion_level(avg_queue: float):
    if avg_queue >= 15:
        return "High"
    if avg_queue >= 8:
        return "Moderate"
    return "Low"


def _filter_rows(rows, term, keys):
    if not term:
        return rows
    t = term.lower().strip()
    out = []
    for row in rows:
        for key in keys:
            if t in str(row.get(key, "")).lower():
                out.append(row)
                break
    return out


def _metric_trend(df: pd.DataFrame, column: str):
    if df is None or column not in df.columns or len(df) < 2:
        return "0", "badge-up"
    prev = _to_int(df.iloc[-2].get(column, 0), 0)
    curr = _to_int(df.iloc[-1].get(column, 0), 0)
    delta = curr - prev
    if delta < 0:
        return f"{delta}", "badge-down"
    return f"+{delta}", "badge-up"


def _active_cameras(df: pd.DataFrame):
    if df is None or "camera_id" not in df.columns:
        return 0
    series = df["camera_id"].dropna().astype(str).str.strip()
    series = series[series != ""]
    return int(series.nunique()) if not series.empty else 0


def _build_camera_rows(df: pd.DataFrame, queue_threshold=10):
    if df is None or "camera_id" not in df.columns:
        return []

    work = df.copy()
    work["camera_id"] = work["camera_id"].astype(str).str.strip()
    work = work[work["camera_id"] != ""]
    if work.empty:
        return []

    sort_col = "frame" if "frame" in work.columns else None
    if sort_col:
        work = work.sort_values(sort_col)

    latest = work.groupby("camera_id", as_index=False).tail(1)

    rows = []
    for _, row in latest.iterrows():
        q = _to_int(row.get("queue_count", 0), 0)
        v = _to_int(row.get("total_vehicles", 0), 0)
        loc = str(row.get("location", "Unknown Location"))
        status = "active" if q < queue_threshold else "warning"
        rows.append(
            {
                "id": str(row.get("camera_id", "Unknown")),
                "location": loc,
                "vehicles": v,
                "queue": q,
                "status": status,
            }
        )

    rows.sort(key=lambda x: x["id"])
    return rows


def _build_recent_violations(df: pd.DataFrame):
    if df is None:
        return []

    work = df.copy()

    if "violation_type" in work.columns:
        events = work[work["violation_type"].notna()].copy()
        events = events[events["violation_type"].astype(str).str.strip() != ""]
        if events.empty:
            return []

        if "frame" in events.columns:
            events = events.sort_values("frame")

        out = []
        for _, row in events.tail(5).iterrows():
            out.append(
                {
                    "type": str(row.get("violation_type", "Violation")),
                    "camera_id": str(row.get("camera_id", "Unknown")),
                    "code": str(row.get("violation_code", "N/A")),
                    "time": _format_time_from_frame(row.get("frame", 0)),
                }
            )
        return out

    red_col = "red_light_violations" if "red_light_violations" in work.columns else None
    rash_col = "rash_driving" if "rash_driving" in work.columns else None
    if red_col is None and rash_col is None:
        return []

    if "frame" in work.columns:
        work = work.sort_values("frame")

    if red_col:
        work["_d_red"] = work[red_col].fillna(0).astype(float).diff().fillna(work[red_col].fillna(0))
    else:
        work["_d_red"] = 0.0

    if rash_col:
        work["_d_rash"] = work[rash_col].fillna(0).astype(float).diff().fillna(work[rash_col].fillna(0))
    else:
        work["_d_rash"] = 0.0

    events = work[(work["_d_red"] > 0) | (work["_d_rash"] > 0)].copy()
    if events.empty:
        return []

    out = []
    for _, row in events.tail(5).iterrows():
        red_delta = _to_int(row.get("_d_red", 0), 0)
        rash_delta = _to_int(row.get("_d_rash", 0), 0)

        if rash_delta > red_delta:
            vtype = "Rash Driving"
            code = "RASH"
        elif red_delta > 0:
            vtype = "Red Light Jump"
            code = "RED"
        else:
            vtype = "Violation"
            code = "GEN"

        out.append(
            {
                "type": vtype,
                "camera_id": str(row.get("camera_id", "Unknown")),
                "code": code,
                "time": _format_time_from_frame(row.get("frame", 0)),
            }
        )

    return out


def show():
    _init_state()

    processed = bool(st.session_state.get("processed", False))
    metrics = st.session_state.get("metrics", {}) or {}
    df = _safe_df()

    total_vehicles = _to_int(metrics.get("total_vehicles", 0), 0)
    queue_count = _to_int(metrics.get("queue_count", 0), 0)
    violations_today = _to_int(metrics.get("red_light_violations", 0), 0)
    active_cameras = _active_cameras(df)

    now = datetime.now()
    if hasattr(st, "autorefresh"):
        try:
            st.autorefresh(interval=1000, key="dash_time_refresh")
        except Exception:
            pass

    trend_active, cls_active = _metric_trend(df, "queue_count")
    trend_vehicles, cls_vehicles = _metric_trend(df, "total_vehicles")
    trend_violations, cls_violations = _metric_trend(df, "red_light_violations")
    trend_queue, cls_queue = _metric_trend(df, "queue_count")

    camera_rows = _build_camera_rows(df)
    violation_rows = _build_recent_violations(df)

    if st.session_state.get("violations"):
        try:
            violations = sorted(
                st.session_state.violations,
                key=lambda x: str(x.get("timestamp", "")),
            )
            violation_rows = [
                {
                    "type": v.get("type", "Violation"),
                    "camera_id": v.get("camera_id", "Unknown"),
                    "code": v.get("violation_id", "N/A"),
                    "time": str(v.get("timestamp", ""))[-8:],
                    "location": v.get("location", "Unknown"),
                }
                for v in violations[-5:]
            ]
        except Exception:
            pass

    search_term = st.session_state.get("dash_search", "").strip()
    if search_term:
        violation_rows = _filter_rows(violation_rows, search_term, ["type", "camera_id", "code", "location"])
        camera_rows = _filter_rows(camera_rows, search_term, ["id", "location", "status"])

    st.markdown(
        """
        <style>
        .dash-root {
            padding: 8px 6px 14px 6px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 18px 20px;
            border-radius: 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
            border: 1px solid #e6edf4;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
            margin-bottom: 16px;
        }

        div[data-testid="stHorizontalBlock"]:has(.top-left) {
            background: #ffffff;
            border: 1px solid #e6edf4;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
            margin-bottom: 16px;
        }

        .top-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .burger {
            font-size: 18px;
            color: #0f172a;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-logo {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            background: linear-gradient(135deg, #14b8a6, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
            font-size: 16px;
            box-shadow: 0 10px 24px rgba(14, 116, 144, 0.25);
        }

        .titleblock h1 {
            margin: 0;
            font-size: 34px;
            line-height: 1.15;
            letter-spacing: -0.5px;
            color: #0f172a;
        }

        .titleblock p {
            margin: 4px 0 0 0;
            color: #5f6b7a;
            font-size: 14px;
        }

        .search-wrap div[data-testid="stTextInput"] input {
            width: 280px;
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-radius: 12px;
            padding: 8px 12px;
            color: #6b7280;
            font-size: 14px;
        }

        .bell {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            background: #ffffff;
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #14b8a6;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 12px;
        }

        .user-chip {
            background: #ffffff;
            border: 1px solid #dbe4ef;
            border-radius: 12px;
            padding: 8px 12px;
            min-width: 230px;
        }

        .user-chip .name {
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
        }

        .user-chip .meta {
            font-size: 12px;
            color: #64748b;
        }

        .flow-grid {
            display: grid;
            gap: 12px;
        }

        .flow-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            font-size: 12px;
            color: #475569;
        }

        .flow-bar {
            flex: 1;
            height: 8px;
            border-radius: 999px;
            background: #e5e7eb;
            overflow: hidden;
            margin: 6px 0 10px 0;
        }

        .flow-fill {
            height: 100%;
            border-radius: 999px;
        }

        .dist-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .dist-card {
            background: #f8fafc;
            border: 1px solid #e9eff6;
            border-radius: 12px;
            padding: 12px;
        }

        .dist-title {
            font-size: 12px;
            color: #64748b;
        }

        .dist-value {
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
        }

        .kpi {
            background: #ffffff;
            border: 1px solid #e6edf4;
            border-radius: 16px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }

        .kpi-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .kpi-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: #0f172a;
            font-size: 13px;
        }
        .kpi-icon svg {
            width: 22px;
            height: 22px;
            stroke: currentColor;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
            fill: none;
        }

        .trend-badge {
            font-size: 11px;
            font-weight: 700;
            border-radius: 999px;
            padding: 4px 8px;
        }

        .badge-up { background: #dcfce7; color: #166534; }
        .badge-down { background: #fee2e2; color: #991b1b; }

        .kpi-label {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 4px;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 34px;
            font-weight: 800;
            line-height: 1;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .panel-title {
            font-size: 26px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        .panel-sub {
            color: #64748b;
            font-size: 13px;
            margin: 4px 0 0 0;
        }

        .section-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .section-head h3 {
            margin: 0;
            color: #0f172a;
            font-size: 24px;
            font-weight: 800;
        }

        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 13px;
        }

        .violation-item {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            background: #f8fafc;
            border: 1px solid #e9eff6;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }

        .vio-left {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .vio-icon {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: #fee2e2;
            color: #b91c1c;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 11px;
        }

        .vio-type {
            margin: 0;
            color: #0f172a;
            font-size: 15px;
            font-weight: 800;
        }

        .vio-loc {
            margin: 0;
            color: #64748b;
            font-size: 12px;
        }

        .vio-right {
            text-align: right;
        }

        .vio-code {
            color: #0f172a;
            font-size: 12px;
            font-weight: 800;
        }

        .vio-time {
            color: #64748b;
            font-size: 12px;
        }

        .cam-item {
            background: #f8fafc;
            border: 1px solid #e9eff6;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 8px;
        }

        .cam-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .dot-green { background: #22c55e; }
        .dot-orange { background: #f97316; }

        .cam-id {
            color: #0f172a;
            font-size: 14px;
            font-weight: 800;
        }

        .cam-status {
            color: #64748b;
            font-size: 12px;
            text-transform: lowercase;
        }

        .cam-loc {
            color: #475569;
            font-size: 12px;
            margin-bottom: 6px;
        }

        .cam-meta {
            display: flex;
            justify-content: space-between;
            color: #334155;
            font-size: 12px;
            font-weight: 600;
        }

        @media (max-width: 1200px) {
            .search-wrap div[data-testid="stTextInput"] input { width: 220px; }
        }

        @media (max-width: 900px) {
            .topbar { flex-direction: column; align-items: flex-start; }
            .search-wrap div[data-testid="stTextInput"] input, .user-chip { width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='dash-root'>", unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.markdown(
            """
            <div class="top-left">
                <span class="burger">☰</span>
                <div class="titleblock">
                    <h1>Traffic Intelligence System</h1>
                    <p>AI-Powered Traffic Analysis &amp; Violation Detection</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown("<div class='top-right'>", unsafe_allow_html=True)
        st.markdown("<div class='search-wrap'>", unsafe_allow_html=True)
        st.text_input("Search", key="dash_search", label_visibility="collapsed", placeholder="Search...")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='bell'>🔔</div>", unsafe_allow_html=True)
        username = st.session_state.get("username", "User")
        role = st.session_state.get("role", "user").title()
        initials = "".join([part[0].upper() for part in str(username).split()[:2]]) or "U"
        st.markdown(
            f"""
            <div class="user-chip">
                <div class="name">{username}</div>
                <div class="meta">{role}</div>
            </div>
            <div class="avatar">{initials}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(
            """
            <div class="panel" style="margin-bottom:14px;">
                <p class="panel-title">Dashboard Overview</p>
                <p class="panel-sub">Real-time traffic monitoring and analytics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:14px; text-align:right;">
                <div class="panel-sub">Current Time</div>
                <div style="font-size:18px;font-weight:800;color:#0f172a;">{now.strftime('%I:%M:%S %p')}</div>
                <div class="panel-sub">{now.strftime('%A, %d %B %Y')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="panel" style="margin-bottom:14px;">', unsafe_allow_html=True)
    st.markdown("### Process Video From Dashboard")
    up_col, btn_col = st.columns([4, 1])
    dash_upload = up_col.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov", "mkv"], key="dashboard_video_upload")
    run_dash = btn_col.button("Process", key="dashboard_process_btn", width="stretch")

    if run_dash:
        if not dash_upload:
            st.warning("Please upload a video before processing.")
        else:
            video_path = _save_uploaded_file(dash_upload)
            st.session_state.last_uploaded_video_path = video_path
            with st.spinner("Processing video..."):
                results = process_full_video(video_path, frame_stride=15, resize_width=520)

            st.session_state.df = results.get("df", pd.DataFrame())
            st.session_state.metrics = results.get("metrics", {})
            st.session_state.run_dir = results.get("run_dir")
            st.session_state.processed_video_path = results.get("output_video_path")
            st.session_state.violations_df = results.get("violations_df", pd.DataFrame())
            st.session_state.processed = True
            st.success("Video processed successfully from dashboard.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi_html = [
        (_lucide_svg("camera"), "#dffaf3", "#0f9f96", "Active Cameras", active_cameras, trend_active, cls_active),
        (_lucide_svg("car"), "#fff4dd", "#ea580c", "Vehicles Detected", total_vehicles, trend_vehicles, cls_vehicles),
        (_lucide_svg("alert-triangle"), "#ffe4e6", "#dc2626", "Violations Today", violations_today, trend_violations, cls_violations),
        (_lucide_svg("map-pin"), "#e8ecff", "#4f46e5", "Avg Queue Length", queue_count, trend_queue, cls_queue),
    ]

    for col, (icon_svg, bg, icon_color, label, value, trend, trend_cls) in zip([c1, c2, c3, c4], kpi_html):
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-top">
                        <div class="kpi-icon" style="background:{bg};color:{icon_color};">{icon_svg}</div>
                        <span class="trend-badge {trend_cls}">{trend}</span>
                    </div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-head">
                    <h3>Recent Violations</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not processed or df is None:
            st.markdown(
                '<div class="info-box">Upload and process video to see violations.</div>',
                unsafe_allow_html=True,
            )
        elif not violation_rows:
            st.markdown(
                '<div class="info-box">No violation events found in current run.</div>',
                unsafe_allow_html=True,
            )
        else:
            for row in violation_rows:
                st.markdown(
                    f"""
                    <div class="violation-item">
                        <div class="vio-left">
                            <div class="vio-icon">AL</div>
                            <div>
                                <p class="vio-type">{row['type']}</p>
                                <p class="vio-loc">{row.get('location', 'Location')} • {row['camera_id']}</p>
                            </div>
                        </div>
                        <div class="vio-right">
                            <div class="vio-code">{row['code']}</div>
                            <div class="vio-time">{row['time']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-head">
                    <h3>Active Cameras</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not processed or df is None:
            st.markdown(
                '<div class="info-box">No active camera data available yet.</div>',
                unsafe_allow_html=True,
            )
        elif not camera_rows:
            st.markdown(
                '<div class="info-box">camera_id column not found in processed data.</div>',
                unsafe_allow_html=True,
            )
        else:
            for cam in camera_rows:
                dot_cls = "dot-green" if cam["status"] == "active" else "dot-orange"
                st.markdown(
                    f"""
                    <div class="cam-item">
                        <div class="cam-top">
                            <div>
                                <span class="status-dot {dot_cls}"></span>
                                <span class="cam-id">{cam['id']}</span>
                            </div>
                            <div class="cam-status">{cam['status']}</div>
                        </div>
                        <div class="cam-loc">{cam['location']}</div>
                        <div class="cam-meta">
                            <span>Vehicles: {cam['vehicles']}</span>
                            <span>Queue: {cam['queue']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if processed and df is not None:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("### Traffic Flow Analysis")

        peak_percent = _traffic_peak_percent(df)
        avg_speed = _avg_speed(df)
        avg_speed_text = f"{avg_speed:.0f} km/h" if avg_speed is not None else "N/A"
        avg_queue = float(df["queue_count"].mean()) if "queue_count" in df.columns and not df.empty else 0.0
        congestion = _congestion_level(avg_queue)
        avg_density = float(df["queue_density"].mean()) if "queue_density" in df.columns and not df.empty else 0.0
        latest_signal = str(df["signal_state"].iloc[-1]) if "signal_state" in df.columns and not df.empty else "N/A"

        st.markdown(
            f"""
            <div class="flow-grid">
                <div class="flow-row">
                    <span>Peak Hour Traffic</span>
                    <span>{peak_percent:.0f}%</span>
                </div>
                <div class="flow-bar"><div class="flow-fill" style="width:{peak_percent:.0f}%;background:#14b8a6;"></div></div>
                <div class="flow-row">
                    <span>Average Speed</span>
                    <span>{avg_speed_text}</span>
                </div>
                <div class="flow-bar"><div class="flow-fill" style="width:{min(100, (avg_speed or 0) * 2):.0f}%;background:#f97316;"></div></div>
                <div class="flow-row">
                    <span>Avg Queue Density</span>
                    <span>{avg_density:.4f}</span>
                </div>
                <div class="flow-bar"><div class="flow-fill" style="width:{min(100, avg_density * 400):.0f}%;background:#22c55e;"></div></div>
                <div class="flow-row">
                    <span>Congestion Level</span>
                    <span>{congestion}</span>
                </div>
                <div class="flow-bar"><div class="flow-fill" style="width:{min(100, avg_queue * 6):.0f}%;background:#f59e0b;"></div></div>
                <div class="flow-row">
                    <span>Signal State</span>
                    <span>{latest_signal}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("### Vehicle Distribution")

        cars = _to_int(metrics.get("cars", 0), 0)
        bikes = _to_int(metrics.get("bikes", 0), 0)
        buses = _to_int(metrics.get("buses", 0), 0)
        trucks = _to_int(metrics.get("trucks", 0), 0)

        st.markdown(
            f"""
            <div class="dist-grid">
                <div class="dist-card" style="background:#ecfeff;">
                    <div class="dist-title">Cars</div>
                    <div class="dist-value">{cars}</div>
                </div>
                <div class="dist-card" style="background:#fff7ed;">
                    <div class="dist-title">Bikes</div>
                    <div class="dist-value">{bikes}</div>
                </div>
                <div class="dist-card" style="background:#eef2ff;">
                    <div class="dist-title">Buses</div>
                    <div class="dist-value">{buses}</div>
                </div>
                <div class="dist-card" style="background:#ffe4e6;">
                    <div class="dist-title">Trucks</div>
                    <div class="dist-value">{trucks}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if processed and df is not None:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("### Traffic Trend Insights")

        cols = [
            col
            for col in ["queue_count", "red_light_violations", "rash_driving", "total_vehicles"]
            if col in df.columns
        ]

        if "frame" in df.columns and cols:
            fig = px.line(df, x="frame", y=cols)
            fig.update_layout(
                height=360,
                margin=dict(l=8, r=8, t=8, b=8),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                legend_title_text="",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(
                '<div class="info-box">Trend chart unavailable for current dataset.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)





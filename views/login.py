import streamlit as st

from backend.auth import create_user, login_user


def _styled_message(message: str, kind: str = "success"):
    css_class = "msg-success" if kind == "success" else "msg-error"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)


def _login_form(role_label: str, username_key: str, password_key: str, button_key: str):
    st.markdown(f"#### {role_label} Login")
    username = st.text_input("Username", key=username_key)
    show_password = st.toggle("Show Password", key=f"{password_key}_show")
    password = st.text_input("Password", type="default" if show_password else "password", key=password_key)

    if st.button(f"Login as {role_label}", key=button_key, width="stretch"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role_label.lower()
            _styled_message(f"{role_label} login successful", kind="success")
            st.rerun()
        else:
            _styled_message("Invalid credentials", kind="error")


def show():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 60%, #e2e8f0 100%);
        }
        [data-testid="stAppViewContainer"] > .main > div {
            min-height: 100vh;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .login-wrap {
            text-align: center;
            margin-bottom: 1rem;
        }
        .login-logo {
            width: 68px;
            height: 68px;
            margin: 0 auto 12px auto;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            color: #ffffff;
            background: linear-gradient(135deg, #14b8a6, #0ea5e9);
            border: 1px solid #e2e8f0;
            box-shadow: 0 16px 35px rgba(15, 23, 42, 0.18);
        }
        .login-title {
            margin: 0;
            font-size: 36px;
            letter-spacing: -0.5px;
            color: #0f172a;
        }
        .login-sub {
            color: #475569;
            margin: 6px 0 0 0;
            font-size: 14px;
        }
        .login-card-anchor + div {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 20px 55px rgba(15, 23, 42, 0.08);
            padding: 36px;
        }
        div[data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 8px;
        }
        div[data-baseweb="tab"] {
            border-radius: 12px !important;
            font-weight: 700 !important;
            background: #f1f5f9 !important;
            color: #0f172a !important;
            border: 1px solid #e2e8f0 !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: #ecfeff !important;
            border-color: #99f6e4 !important;
            color: #0f172a !important;
        }
        h4 {
            color: #0f172a !important;
            margin-bottom: 0.7rem !important;
        }
        label[data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
            font-weight: 600 !important;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            padding: 0.65rem 0.8rem !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #14b8a6 !important;
            box-shadow: 0 0 0 0.18rem rgba(20, 184, 166, 0.2) !important;
        }
        div[data-testid="stToggle"] label p {
            color: #475569 !important;
            font-size: 14px;
        }
        div[data-testid="stToggle"] {
            margin-top: 4px;
            margin-bottom: 8px;
        }
        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid #14b8a6 !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #0ea5e9, #14b8a6) !important;
            font-weight: 700 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.18) !important;
            min-height: 2.75rem !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(12, 39, 54, 0.42) !important;
            filter: brightness(1.06);
        }
        .stButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(134, 210, 255, 0.35) !important;
        }
        .msg-error, .msg-success {
            border-radius: 12px;
            padding: 11px 14px;
            font-weight: 600;
            margin-top: 10px;
            border: 1px solid;
        }
        .msg-error {
            color: #ffe2e2;
            background: rgba(143, 24, 24, 0.42);
            border-color: rgba(255, 120, 120, 0.6);
        }
        .msg-success {
            color: #dcffe9;
            background: rgba(17, 102, 62, 0.42);
            border-color: rgba(103, 255, 179, 0.55);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-logo">TA</div>
            <h1 class="login-title">TrafficAI Login</h1>
            <p class="login-sub">Sign in as Admin or User to access traffic intelligence modules.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card-anchor"></div>', unsafe_allow_html=True)
        tab_admin, tab_user, tab_signup = st.tabs(["Admin Login", "User Login", "Create Account"])

        with tab_admin:
            _login_form("Admin", "admin_login_user", "admin_login_pass", "admin_login_btn")

        with tab_user:
            _login_form("User", "user_login_user", "user_login_pass", "user_login_btn")

        with tab_signup:
            st.markdown("#### Register New User")
            new_user = st.text_input("Create Username", key="signup_user")
            show_signup_password = st.toggle("?? Show Password", key="signup_pass_show")
            new_pass = st.text_input(
                "Create Password",
                type="default" if show_signup_password else "password",
                key="signup_pass",
            )

            if st.button("Create Account", key="signup_btn", width="stretch"):
                created = create_user(new_user, new_pass)
                if created:
                    _styled_message("Account created. You can now log in from User or Admin tab.", kind="success")
                else:
                    _styled_message("Username already exists", kind="error")

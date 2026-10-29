# Name : Kunal Machhindra Vadkte
# Class : B.Tech IT B (T3B)
# PRN : UIT23M1134
# Assignment 12: Industry Capstone – End-to-End NLP Applicationss

"""
app.py
======
Scholarship Document Summarizer
Industry-grade NLP web application with Login / Signup authentication.
"""

import sys, os, time, json, hashlib, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScholarAI – Scholarship Document Summarizer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from modules.pdf_extractor import extract_text
from modules.preprocessing import preprocess_text, clean_text_for_display
from modules.information_extractor import extract_all
from modules.summarizer import summarize
from modules.eligibility_checker import check_eligibility

# ---------------------------------------------------------------------------
# Simple File-Based User Store (JSON in data/)
# ---------------------------------------------------------------------------
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.json")

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def auth_signup(name: str, email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    if not name or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."
    users[email] = {"name": name.strip(), "password": _hash(password)}
    _save_users(users)
    return True, "Account created successfully!"

def auth_login(email: str, password: str) -> tuple[bool, str, str]:
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        return False, "No account found with this email.", ""
    if users[email]["password"] != _hash(password):
        return False, "Incorrect password. Please try again.", ""
    return True, "Welcome back!", users[email]["name"]

# ---------------------------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------------------------
defaults = {
    "auth_page": "login",   # "login" | "signup" | "app"
    "logged_in": False,
    "user_name": "",
    "user_email": "",
    "analysis_done": False,
    "extracted_text": "",
    "info": {},
    "summary_result": {},
    "preprocess_result": {},
    "filename": "",
    "pages": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# ── GLOBAL CSS (High Contrast, Professional Industry Theme) ────────────────
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Fonts ──────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root & Canvas ───────────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: #090d16 !important;
        color: #f1f5f9 !important;
        font-size: 15px;
    }
    .main, .block-container {
        background: transparent !important;
    }
    .block-container {
        padding: 1.75rem 2rem 4rem 2rem !important;
        max-width: 1200px !important;
    }

    /* ── Hide Streamlit chrome ─────────────────────────────── */
    #MainMenu, footer, header, .stDeployButton,
    div[data-testid="stToolbar"], div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ══════════════════════════════════════════════════════════
       ── NAVIGATION BAR ───────────────────────────────────────
    ══════════════════════════════════════════════════════════ */
    .s-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.25rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .s-logo {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .s-logo-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem;
        box-shadow: 0 0 16px rgba(99,102,241,0.5);
    }
    .s-logo-text {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .s-logo-text span { color: #818cf8; }
    .s-nav-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .s-nav-badge {
        background: rgba(99,102,241,0.18);
        border: 1px solid rgba(129,140,248,0.4);
        color: #c7d2fe;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .s-user-dot {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem; color: #ffffff;
        box-shadow: 0 0 10px rgba(99,102,241,0.4);
    }

    /* ══════════════════════════════════════════════════════════
       ── AUTH FORMS (Card Box & High Visibility Inputs) ────────
    ══════════════════════════════════════════════════════════ */
    div[data-testid="stForm"] {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 18px !important;
        padding: 2.2rem 2.2rem !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6), 0 0 35px rgba(99, 102, 241, 0.12) !important;
    }

    /* Form labels - clearly visible light slate */
    label, .stTextInput label, .stSelectbox label, .stSlider label, .stFileUploader label {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.4rem !important;
    }

    /* Hide Streamlit's "Press Enter to submit form" floating instructions */
    div[data-testid="InputInstructions"],
    div[data-testid="stTextInputRootElement"] div[data-testid="InputInstructions"],
    .stTextInput div[data-testid="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Reset all outer containers so no nested double borders occur */
    div[data-testid="stTextInputRootElement"],
    .stTextInput > div,
    .stTextInput > div > div,
    div[data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Single border container for inputs */
    div[data-baseweb="input"] {
        background-color: #1e293b !important;
        border: 1.5px solid #334155 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.35) !important;
        background-color: #1e293b !important;
    }

    /* Actual text input */
    div[data-baseweb="input"] input,
    input {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        caret-color: #818cf8 !important;
        padding: 0.65rem 0.85rem !important;
    }

    /* Placeholder text */
    input::placeholder,
    div[data-baseweb="input"] input::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* Password reveal eye button */
    div[data-baseweb="input"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        padding: 0 0.75rem !important;
        cursor: pointer !important;
    }
    div[data-baseweb="input"] button:hover {
        background: transparent !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="input"] button svg {
        color: #94a3b8 !important;
        fill: currentColor !important;
    }

    /* ══════════════════════════════════════════════════════════
       ── BUTTONS (High Contrast & Tactile) ───────────────────
    ══════════════════════════════════════════════════════════ */
    /* Primary / Form Submit buttons */
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: 1px solid #818cf8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.65) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stFormSubmitButton"] > button *,
    button[kind="primary"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* Secondary action buttons */
    .stButton > button {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 1.5px solid #475569 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.5rem !important;
        cursor: pointer !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #334155 !important;
        border-color: #818cf8 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important;
    }
    .stButton > button * {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover * {
        color: #ffffff !important;
    }

    /* ══════════════════════════════════════════════════════════
       ── UPLOAD ZONE ─────────────────────────────────────────
    ══════════════════════════════════════════════════════════ */
    div[data-testid="stFileUploader"] {
        background: #0f172a !important;
        border: 2px dashed rgba(99,102,241,0.4) !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(99,102,241,0.75) !important;
    }

    /* ══════════════════════════════════════════════════════════
       ── TABS ────────────────────────────────────────────────
    ══════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: #0f172a !important;
        border-radius: 12px !important;
        padding: 6px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background: rgba(255,255,255,0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    }

    /* ══════════════════════════════════════════════════════════
       ── HERO SECTION ────────────────────────────────────────
    ══════════════════════════════════════════════════════════ */
    .hero-section {
        text-align: center;
        padding: 2rem 1rem 2.5rem 1rem;
        max-width: 720px;
        margin: 0 auto;
    }
    .hero-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.35);
        color: #c7d2fe;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 1.2rem;
    }
    .hero-h1 {
        font-size: 2.9rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0 0 1rem 0;
    }
    .hero-h1 .grad {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-p {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.7;
        margin: 0;
    }

    /* ══════════════════════════════════════════════════════════
       ── CARDS & METRIC WIDGETS ──────────────────────────────
    ══════════════════════════════════════════════════════════ */
    .s-card {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .s-card:hover {
        border-color: rgba(129,140,248,0.3);
        transform: translateY(-1px);
    }
    .s-card-accent {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .s-card-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .s-card-value {
        font-size: 1.08rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.5;
        word-break: break-word;
    }
    .s-card-value-muted {
        font-size: 0.95rem;
        color: #64748b;
        font-style: italic;
    }

    /* Stat strip */
    .stat-strip {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .stat-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Section Headings */
    .s-heading {
        font-size: 1.12rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 1.75rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }

    /* Doc tags */
    .doc-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(52, 211, 153, 0.35);
        color: #6ee7b7;
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    /* Eligibility Rows & Badges */
    .elig-row {
        background: #0f172a;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
    }
    .badge-met {
        background: rgba(34,197,94,0.18);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-notmet {
        background: rgba(239,68,68,0.18);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-unk {
        background: rgba(245,158,11,0.18);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Summary block */
    .summary-block {
        background: #0f172a;
        border: 1px solid rgba(99,102,241,0.3);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        font-size: 1.02rem;
        line-height: 1.85;
        color: #e2e8f0;
    }

    /* Text area */
    .stTextArea textarea {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        font-size: 0.92rem !important;
        line-height: 1.65 !important;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background: #0f172a !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Dataframe */
    .stDataFrame {
        background: #0f172a !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #0f172a !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# ── HELPERS ─────────────────────────────────────────────────────────────────
# ===========================================================================
def navbar(show_user: bool = False):
    initials = ""
    if show_user and st.session_state.user_name:
        initials = "".join(w[0] for w in st.session_state.user_name.split()[:2]).upper()

    user_html = f'<span style="font-size:0.9rem;color:#cbd5e1;font-weight:600;">{st.session_state.user_name}</span><div class="s-user-dot">{initials}</div>' if (show_user and initials) else ""

    st.markdown(
        f'<div class="s-navbar">'
        f'<div class="s-logo">'
        f'<div class="s-logo-icon">🎓</div>'
        f'<div class="s-logo-text">Scholar<span>AI</span></div>'
        f'</div>'
        f'<div class="s-nav-right">'
        f'<span class="s-nav-badge">NLP Intelligence</span>'
        f'{user_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def s_card(label: str, value: str, icon: str, accent: str):
    v = value if value and "not found" not in str(value).lower() else None
    val_html = f'<div class="s-card-value">{v}</div>' if v else '<div class="s-card-value-muted">Not identified in document</div>'
    st.markdown(
        f'<div class="s-card">'
        f'<div class="s-card-accent" style="background:{accent};"></div>'
        f'<div class="s-card-label">{icon} {label}</div>'
        f'{val_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def _ensure_demo():
    """Ensure demo account always exists."""
    users = _load_users()
    if "demo@scholarai.com" not in users:
        users["demo@scholarai.com"] = {"name": "Demo User", "password": _hash("demo123")}
        _save_users(users)


# ===========================================================================
# ── AUTH: LOGIN PAGE ─────────────────────────────────────────────────────────
# ===========================================================================
def page_login():
    navbar()

    # Centered card layout
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; background:linear-gradient(135deg, #6366f1, #a855f7); border-radius:12px; font-size:1.5rem; margin-bottom:0.75rem; box-shadow:0 0 20px rgba(99,102,241,0.5);">🎓</div>
            <div style="font-size:1.85rem; font-weight:800; color:#ffffff; letter-spacing:-0.025em;">Welcome back</div>
            <div style="font-size:0.95rem; color:#94a3b8; margin-top:0.3rem;">Sign in to your ScholarAI workspace</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="e.g. demo@scholarai.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in →", use_container_width=True)

        if submitted:
            ok, msg, name = auth_login(email, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.session_state.user_email = email.strip().lower()
                st.session_state.auth_page = "app"
                st.success(f"Welcome back, {name}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

        # Demo credentials helper box
        st.markdown("""
        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(129,140,248,0.3); border-radius:10px; padding:0.85rem 1rem; margin-top:1.2rem; text-align:center;">
            <div style="font-size:0.8rem; font-weight:700; color:#c7d2fe; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">Instant Demo Access</div>
            <div style="font-size:0.88rem; color:#e2e8f0;">
                Email: <code style="color:#a5b4fc; background:rgba(0,0,0,0.35); padding:0.15rem 0.45rem; border-radius:5px; font-weight:600;">demo@scholarai.com</code> &nbsp;·&nbsp;
                Password: <code style="color:#a5b4fc; background:rgba(0,0,0,0.35); padding:0.15rem 0.45rem; border-radius:5px; font-weight:600;">demo123</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:1.2rem; text-align:center; font-size:0.92rem; color:#94a3b8;'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Create a free account →", use_container_width=True):
            _ensure_demo()
            st.session_state.auth_page = "signup"
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:2.5rem; font-size:0.85rem; color:#64748b;">
        ScholarAI NLP Platform · Secure Authentication & Document Intelligence
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# ── AUTH: SIGNUP PAGE ────────────────────────────────────────────────────────
# ===========================================================================
def page_signup():
    navbar()

    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; background:linear-gradient(135deg, #6366f1, #a855f7); border-radius:12px; font-size:1.5rem; margin-bottom:0.75rem; box-shadow:0 0 20px rgba(99,102,241,0.5);">🎓</div>
            <div style="font-size:1.85rem; font-weight:800; color:#ffffff; letter-spacing:-0.025em;">Create your account</div>
            <div style="font-size:0.95rem; color:#94a3b8; margin-top:0.3rem;">Start analyzing scholarship circulars in seconds</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("signup_form"):
            name = st.text_input("Full name", placeholder="e.g. Arjun Sharma")
            email = st.text_input("Email address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="At least 6 characters")
            confirm = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
            submitted = st.form_submit_button("Create account →", use_container_width=True)

        if submitted:
            if password != confirm:
                st.error("Passwords do not match. Please re-enter.")
            else:
                ok, msg = auth_signup(name, email, password)
                if ok:
                    st.success(f"{msg} You can now log in.")
                    time.sleep(1)
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("<div style='margin-top:1.5rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:1.2rem; text-align:center; font-size:0.92rem; color:#94a3b8;'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Sign in instead →", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:2.5rem; font-size:0.85rem; color:#64748b;">
        By creating an account, you agree to the Terms of Service and Privacy Policy.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# ── MAIN APPLICATION ─────────────────────────────────────────────────────────
# ===========================================================================
def page_app():
    # Topbar with user profile and logout
    c_logo, c_spacer, c_out = st.columns([4, 4, 3])
    with c_logo:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.65rem; padding:0.25rem 0 1rem 0; border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:0.5rem;">
            <div class="s-logo-icon">🎓</div>
            <div class="s-logo-text">Scholar<span style="color:#818cf8;">AI</span></div>
            <span class="s-nav-badge" style="margin-left:0.5rem;">Document Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

    with c_out:
        col_a, col_b = st.columns([1.6, 1])
        with col_a:
            st.markdown(f"<div style='font-size:0.92rem; color:#cbd5e1; font-weight:600; padding-top:0.5rem; text-align:right;'>👤 {st.session_state.user_name}</div>", unsafe_allow_html=True)
        with col_b:
            if st.button("Sign out"):
                for k in ["logged_in", "user_name", "user_email", "analysis_done",
                          "extracted_text", "info", "summary_result", "preprocess_result",
                          "filename", "pages"]:
                    st.session_state[k] = defaults[k]
                st.session_state.auth_page = "login"
                st.rerun()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-section">
        <div class="hero-label">✦ NLP Pipeline · NLTK · spaCy · TF-IDF</div>
        <h1 class="hero-h1">Scholarship Document<br><span class="grad">Summarizer</span></h1>
        <p class="hero-p">Upload any scholarship circular or PDF notice. Our NLP pipeline automatically extracts key criteria, generates an executive summary, and validates your eligibility.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + Config Row ───────────────────────────────────────────────────
    col_left, col_right = st.columns([5, 3], gap="large")

    with col_left:
        t_upload, t_demo = st.tabs(["📄 Upload Document", "⚡ Try Demo"])

        uploaded_pdf = None
        demo_selected = False

        with t_upload:
            uploaded_pdf = st.file_uploader(
                "PDF document",
                type=["pdf"],
                key="pdf_uploader",
                label_visibility="collapsed",
            )
            if uploaded_pdf:
                st.success(f"✓ **{uploaded_pdf.name}** — {uploaded_pdf.size / 1024:.1f} KB ready to analyze.")

        with t_demo:
            st.markdown("""
            <div style="background:#0f172a; border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
                <div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:0.4rem;">ABC Education Foundation Scholarship 2026</div>
                <div style="color:#94a3b8; font-size:0.88rem; line-height:1.6;">
                    Official sample notice featuring ₹50,000 annual award, 75% academic marks cut-off, ₹4 Lakh income ceiling, and 9 required documents — ready to test all 5 pipeline stages.
                </div>
            </div>
            """, unsafe_allow_html=True)
            demo_btn = st.button("Load Sample Document", use_container_width=True)
            if demo_btn:
                demo_selected = True

        can_run = (uploaded_pdf is not None) or demo_selected
        run_btn = st.button(
            "🚀 Analyze Document",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

    with col_right:
        summary_n = st.slider("Summary length (sentences)", 2, 8, 4)
        st.markdown("""
        <div style="background:#0f172a; border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:1.2rem; margin-top:0.75rem;">
            <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#94a3b8; margin-bottom:0.75rem;">Pipeline Processing Steps</div>
            <div style="display:flex; flex-direction:column; gap:0.55rem;">
                <div style="display:flex; align-items:center; gap:0.65rem; font-size:0.88rem; color:#cbd5e1;"><span style="width:22px; height:22px; border-radius:50%; background:rgba(99,102,241,0.25); border:1px solid rgba(129,140,248,0.4); display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; color:#818cf8;">1</span>PDF text extraction (PyMuPDF)</div>
                <div style="display:flex; align-items:center; gap:0.65rem; font-size:0.88rem; color:#cbd5e1;"><span style="width:22px; height:22px; border-radius:50%; background:rgba(99,102,241,0.25); border:1px solid rgba(129,140,248,0.4); display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; color:#818cf8;">2</span>Tokenization & lemmatization (NLTK)</div>
                <div style="display:flex; align-items:center; gap:0.65rem; font-size:0.88rem; color:#cbd5e1;"><span style="width:22px; height:22px; border-radius:50%; background:rgba(99,102,241,0.25); border:1px solid rgba(129,140,248,0.4); display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; color:#818cf8;">3</span>Entity extraction — Regex + spaCy NER</div>
                <div style="display:flex; align-items:center; gap:0.65rem; font-size:0.88rem; color:#cbd5e1;"><span style="width:22px; height:22px; border-radius:50%; background:rgba(99,102,241,0.25); border:1px solid rgba(129,140,248,0.4); display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; color:#818cf8;">4</span>TF-IDF extractive summarization</div>
                <div style="display:flex; align-items:center; gap:0.65rem; font-size:0.88rem; color:#cbd5e1;"><span style="width:22px; height:22px; border-radius:50%; background:rgba(99,102,241,0.25); border:1px solid rgba(129,140,248,0.4); display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; color:#818cf8;">5</span>Automated eligibility qualification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline Execution ────────────────────────────────────────────────────
    if run_btn or demo_selected:
        with st.spinner("Processing document through NLP pipeline…"):
            pb = st.progress(0)
            pb.progress(15, text="Extracting text from PDF…")
            if demo_selected or (uploaded_pdf is None and os.path.exists("data/sample_scholarship.txt")):
                with open("data/sample_scholarship.txt", "r", encoding="utf-8") as f:
                    raw_text = f.read()
                doc_name, doc_pages = "ABC_Foundation_Scholarship_2026 (Demo)", 2
            else:
                res = extract_text(uploaded_pdf)
                if not res["success"]:
                    st.error(f"Extraction failed: {res['error']}")
                    st.stop()
                raw_text, doc_name, doc_pages = res["text"], uploaded_pdf.name, res["pages"]

            pb.progress(40, text="Tokenizing and lemmatizing text…")
            prep_res = preprocess_text(raw_text)
            pb.progress(65, text="Extracting entities (spaCy + Regex)…")
            info = extract_all(raw_text)
            pb.progress(85, text="Ranking sentences via TF-IDF…")
            summ = summarize(raw_text, num_sentences=summary_n)
            pb.progress(100, text="Complete!")
            time.sleep(0.25)
            pb.empty()

            st.session_state.update({
                "analysis_done": True,
                "extracted_text": raw_text,
                "info": info,
                "summary_result": summ,
                "preprocess_result": prep_res,
                "filename": doc_name,
                "pages": doc_pages,
            })
        st.toast("Analysis complete!", icon="✅")

    # ── Results Dashboard ─────────────────────────────────────────────────────
    if not st.session_state.analysis_done:
        return

    info = st.session_state.info
    summ = st.session_state.summary_result
    prep = st.session_state.preprocess_result
    raw  = st.session_state.extracted_text

    total_words   = prep.get("word_count", len(raw.split()))
    total_sents   = prep.get("sentence_count", 0)
    docs_count    = len(info.get("documents", []))
    summ_words    = len(summ.get("summary", "").split())
    compression   = max(0, int((1 - summ_words / max(1, total_words)) * 100))

    st.markdown("<hr style='margin: 2.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

    # ── Stats row ────────────────────────────────────────────────────────────
    sa, sb, sc, sd, se = st.columns(5)
    for col, val, lbl in [
        (sa, st.session_state.pages, "Pages"),
        (sb, total_words, "Total Words"),
        (sc, total_sents, "Sentences"),
        (sd, docs_count, "Req. Documents"),
        (se, f"{compression}%", "Compression"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-strip">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📋 Key Information",
        "🧠 Executive Summary",
        "🎯 Eligibility Checker",
        "🔬 NLP Diagnostics",
        "📄 Full Document Text",
    ])

    # ── Tab 1: Key Information ────────────────────────────────────────────────
    with t1:
        st.markdown('<div class="s-heading">📌 Extracted Scholarship Metadata</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            s_card("Scholarship Name",   info.get("scholarship_name"),         "📌", "linear-gradient(90deg,#6366f1,#818cf8)")
            s_card("Academic Marks / CGPA", info.get("percentage_requirement"), "📈", "linear-gradient(90deg,#f59e0b,#fbbf24)")
        with r1c2:
            s_card("Award Amount",       info.get("amount"),                   "💰", "linear-gradient(90deg,#10b981,#34d399)")
            s_card("Income Limit",       info.get("income_requirement"),       "💵", "linear-gradient(90deg,#f43f5e,#fb7185)")
        with r1c3:
            s_card("Issuing Organization", info.get("organization"),           "🏢", "linear-gradient(90deg,#38bdf8,#60a5fa)")
            s_card("Application Deadline", info.get("deadline"),               "📅", "linear-gradient(90deg,#a855f7,#c084fc)")

        st.markdown('<div class="s-heading">📁 Mandatory Documentation</div>', unsafe_allow_html=True)
        docs = info.get("documents", [])
        if docs:
            html = "".join(f'<span class="doc-tag">✓ {d}</span>' for d in docs)
            st.markdown(f'<div style="margin-bottom:1.5rem;">{html}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific documents were identified in this document.")

        ap_col, el_col = st.columns(2)
        with ap_col:
            st.markdown('<div class="s-heading">📝 Application Procedure</div>', unsafe_allow_html=True)
            ap = info.get("application_process", "")
            if ap and "not found" not in ap.lower():
                st.markdown(f'<div class="s-card" style="font-size:0.93rem; line-height:1.75; color:#cbd5e1;">{ap.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
            else:
                st.info("Application procedure not detected in this document.")

        with el_col:
            st.markdown('<div class="s-heading">✅ Eligibility Criteria</div>', unsafe_allow_html=True)
            el = info.get("eligibility", "")
            if el and "not found" not in el.lower():
                st.markdown(f'<div class="s-card" style="font-size:0.93rem; line-height:1.75; color:#cbd5e1;">{el.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
            else:
                st.info("Eligibility paragraph not detected in this document.")

    # ── Tab 2: Summary ────────────────────────────────────────────────────────
    with t2:
        st.markdown('<div class="s-heading">🧠 TF-IDF Extractive Summary</div>', unsafe_allow_html=True)
        if summ.get("success") and summ.get("summary"):
            st.markdown(f"""
            <div class="summary-block">
                <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:#818cf8; margin-bottom:0.75rem;">
                    Key Sentences Extracted ({summ.get("sentence_count", 0)} sentences)
                </div>
                {summ["summary"]}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.86rem; color:#94a3b8; margin-top:0.75rem;">Compressed from {total_words} words to {summ_words} words ({compression}% reduction).</div>', unsafe_allow_html=True)

            scored = summ.get("scored_sentences", [])
            if scored:
                with st.expander("View sentence importance rankings (TF-IDF scores)"):
                    for rank, (sent, score) in enumerate(sorted(scored, key=lambda x: x[1], reverse=True)[:8], 1):
                        st.markdown(f"""
                        <div style="background:#0f172a; border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:0.85rem 1rem; margin-bottom:0.65rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                                <span style="font-size:0.8rem; font-weight:700; color:#818cf8;">Rank #{rank}</span>
                                <span style="background:rgba(99,102,241,0.18); color:#c7d2fe; font-size:0.76rem; font-weight:700; padding:0.15rem 0.6rem; border-radius:9999px; border:1px solid rgba(129,140,248,0.35);">Score: {score:.4f}</span>
                            </div>
                            <div style="font-size:0.92rem; color:#e2e8f0; line-height:1.6;">{sent}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Summary could not be generated for this document.")

    # ── Tab 3: Eligibility Check ──────────────────────────────────────────────
    with t3:
        st.markdown('<div class="s-heading">🎯 Eligibility Verification</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.95rem; color:#94a3b8; margin-bottom:1.4rem;">Enter your academic and financial criteria to check eligibility against this circular.</div>', unsafe_allow_html=True)

        with st.form("elig_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st_pct = st.text_input("Academic Marks / Percentage (%)", placeholder="e.g. 82.5")
            with fc2:
                st_inc = st.text_input("Annual Family Income (₹)", placeholder="e.g. 350000")
            with fc3:
                st_crs = st.text_input("Current Degree / Program", placeholder="e.g. B.Tech")
            check_btn = st.form_submit_button("Evaluate My Eligibility", type="primary", use_container_width=True)

        if check_btn:
            elig = check_eligibility({"percentage": st_pct, "income": st_inc, "course": st_crs}, info)
            verdict = elig.get("overall", "insufficient_data")
            if verdict == "likely_eligible":
                st.success(f"🎉 {elig.get('summary')}")
            elif verdict == "likely_ineligible":
                st.error(f"⚠️ {elig.get('summary')}")
            else:
                st.warning(f"ℹ️ {elig.get('summary')}")

            for r in elig.get("results", []):
                s = r["status"]
                badge = (f'<span class="badge-met">Eligible</span>' if s == "met"
                         else f'<span class="badge-notmet">Not Eligible</span>' if s == "not_met"
                         else f'<span class="badge-unk">Review</span>')
                st.markdown(f"""
                <div class="elig-row">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f8fafc; font-size:0.98rem;">{r["field"]}</span>
                        {badge}
                    </div>
                    <div style="font-size:0.86rem; color:#94a3b8; margin-top:0.35rem;">
                        Required: <strong style="color:#cbd5e1;">{r["required"]}</strong> &nbsp;·&nbsp; Yours: <strong style="color:#cbd5e1;">{r["provided"]}</strong>
                    </div>
                    <div style="font-size:0.92rem; color:#cbd5e1; margin-top:0.4rem; line-height:1.5;">{r["message"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: NLP Diagnostics ────────────────────────────────────────────────
    with t4:
        st.markdown('<div class="s-heading">🔬 NLP Pipeline Diagnostics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Words", prep.get("word_count", 0))
        d2.metric("Total Sentences", prep.get("sentence_count", 0))
        tokens = prep.get("clean_tokens", [])
        d3.metric("Cleaned Tokens", len(tokens))
        d4.metric("Unique Lemmas", len(set(tokens)))

        if tokens:
            st.markdown(f"""<div style="background:#0f172a; border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:1.1rem; margin:1.2rem 0;">
                <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#94a3b8; margin-bottom:0.6rem;">Sample Processed Tokens (First 40)</div>
                <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.8;">{"  ·  ".join(tokens[:40])}</div>
            </div>""", unsafe_allow_html=True)

            counts = Counter(tokens).most_common(10)
            df = pd.DataFrame(counts, columns=["Token", "Frequency Count"])
            st.markdown('<div style="font-size:0.88rem; font-weight:700; color:#cbd5e1; margin-bottom:0.5rem;">Top 10 Most Frequent Keywords</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 5: Raw Text ───────────────────────────────────────────────────────
    with t5:
        st.markdown('<div class="s-heading">📄 Extracted Document Text</div>', unsafe_allow_html=True)
        st.text_area("", value=clean_text_for_display(raw), height=400, disabled=True, label_visibility="collapsed")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<hr style='margin-top:3.5rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:1rem 0; flex-wrap:wrap; gap:0.75rem;">
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <div style="background:linear-gradient(135deg,#6366f1,#a855f7); width:26px; height:26px; border-radius:7px; display:flex; align-items:center; justify-content:center; font-size:0.8rem;">🎓</div>
            <span style="font-weight:700; color:#ffffff; font-size:0.92rem;">ScholarAI</span>
            <span style="color:#64748b; font-size:0.85rem;"> · NLP Intelligence Platform</span>
        </div>
        <div style="font-size:0.85rem; color:#64748b;">
            Python · Streamlit · NLTK · spaCy · scikit-learn · PyMuPDF
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# ── ROUTER ───────────────────────────────────────────────────────────────────
# ===========================================================================
# Ensure demo account exists on every load
_ensure_demo()

if not st.session_state.logged_in:
    if st.session_state.auth_page == "signup":
        page_signup()
    else:
        page_login()
else:
    st.session_state.auth_page = "app"
    page_app()

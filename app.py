import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Trends Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* White background + prevent horizontal scroll */
.stApp {
    background-color: #ffffff;
    overflow-x: hidden !important;
}

/* Main content area compact */
.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}

/* Prevent any element from overflowing */
* {
    box-sizing: border-box;
}

/* Charts fill column width */
.js-plotly-plot, .plotly {
    width: 100% !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 8px 0 !important;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(102,126,234,0.25);
    transition: transform 0.2s;
    color: white;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: white;
    margin: 6px 0 4px;
    word-break: break-word;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
    letter-spacing: 0.5px;
}
.metric-card .icon { font-size: 2rem; }

/* Green metric */
.metric-card-green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 8px 32px rgba(17,153,142,0.3);
}
/* Orange metric */
.metric-card-orange {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 8px 32px rgba(245,87,108,0.3);
}
/* Blue metric */
.metric-card-blue {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    box-shadow: 0 8px 32px rgba(79,172,254,0.3);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 20px;
    padding: 32px 28px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(102,126,234,0.15) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.75);
    font-weight: 400;
    margin-bottom: 0;
}

/* Section headers */
.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 6px;
}
.section-sub {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 28px;
}

/* Nav buttons */
.nav-btn {
    display: block;
    width: 100%;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    text-decoration: none;
    margin-bottom: 12px;
}

/* Input form card */
.form-card {
    background: #f8faff;
    border-radius: 20px;
    padding: 32px;
    border: 1px solid #e8edf5;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

/* Result card - growing */
.result-growing {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(17,153,142,0.35);
    color: white;
}
/* Result card - shrinking */
.result-shrinking {
    background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(245,87,108,0.35);
    color: white;
}
/* Result card - salary */
.result-salary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(102,126,234,0.35);
    color: white;
}
.result-title {
    font-size: 1.1rem;
    font-weight: 500;
    opacity: 0.9;
    margin-bottom: 8px;
}
.result-value {
    font-size: 3rem;
    font-weight: 800;
    margin: 8px 0;
}
.result-desc {
    font-size: 1rem;
    opacity: 0.85;
    margin-top: 8px;
}

/* Cluster card */
.cluster-card {
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

/* Divider */
.custom-divider {
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    border-radius: 2px;
    margin: 24px 0;
}

/* Page title strip */
.page-title-strip {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.page-title-strip .pt-icon { font-size: 2.2rem; }
.page-title-strip .pt-text h2 {
    color: white;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}
.page-title-strip .pt-text p {
    color: rgba(255,255,255,0.7);
    margin: 4px 0 0;
    font-size: 0.95rem;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(102,126,234,0.5);
}

/* Slider label */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 0.95rem !important;
}

/* About card */
.about-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    border: 1px solid #e8edf5;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.about-card h3 {
    color: #1a1a2e;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Sidebar logo area */
.sidebar-logo {
    text-align: center;
    padding: 20px 10px 30px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}
.sidebar-logo h2 {
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 10px 0 4px;
}
.sidebar-logo p {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING & MODEL TRAINING ──────────────────────────────────────────
import os as _os

CSV_NAME = 'ai_in_job_treands_without_skill.csv'

def _find_csv():
    import sys

    # All candidate locations to search
    candidates = []

    # 1. Current working directory (most reliable with streamlit run)
    candidates.append(_os.getcwd())

    # 2. Directory of this script via __file__
    try:
        candidates.append(_os.path.dirname(_os.path.abspath(__file__)))
    except Exception:
        pass

    # 3. Directory of the first sys.argv entry (the script path)
    try:
        if sys.argv and sys.argv[0]:
            candidates.append(_os.path.dirname(_os.path.abspath(sys.argv[0])))
    except Exception:
        pass

    # 4. Any path containing app.py in sys.path
    for p in sys.path:
        candidates.append(p)

    for folder in candidates:
        full = _os.path.join(folder, CSV_NAME)
        if _os.path.isfile(full):
            return full

    return None

@st.cache_data
def load_data(csv_path=None):
    if csv_path:
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def train_models(df):
    df_enc = df.copy()
    le_dict = {}
    cat_cols = ['Industry', 'AI_Impact_Level', 'Required_Education', 'Location', 'Job_Status']
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # ID3 Classification - Job Status
    cls_features = ['Industry','AI_Impact_Level','Median_Salary_USD','Experience_Required_Years',
                    'Job_Openings_2024','Projected_Openings_2030','Remote_Work_Ratio_',
                    'Automation_Risk_','Location','Required_Education','Gender_Diversity_']
    X_cls = df_enc[cls_features]
    y_cls = df_enc['Job_Status']
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42)
    dt.fit(X_cls, y_cls)

    # MLR - Salary
    reg_features = ['Industry','AI_Impact_Level','Experience_Required_Years',
                    'Job_Openings_2024','Projected_Openings_2030','Remote_Work_Ratio_',
                    'Automation_Risk_','Location','Required_Education','Gender_Diversity_']
    X_reg = df_enc[reg_features]
    y_reg = df['Median_Salary_USD']
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)

    # K-Means
    km_features = ['Median_Salary_USD','Automation_Risk_','Experience_Required_Years','Remote_Work_Ratio_']
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(df_enc[km_features])

    return dt, lr, km, le_dict, cls_features, reg_features, km_features

# ── Try to find CSV automatically first ──────────────────────────────────────
_auto_path = _find_csv()

# Debug info shown briefly in sidebar (remove after confirming it works)
if _auto_path is None:
    import sys, os as _dbos
    _debug_cwd = _dbos.getcwd()
    _debug_files = _dbos.listdir(_debug_cwd) if _dbos.path.exists(_debug_cwd) else []

if _auto_path:
    df = load_data(_auto_path)
else:
    # ── Friendly full-page uploader ───────────────────────────────────────────
    import os as _uos
    _cwd_now = _uos.getcwd()
    _files_now = _uos.listdir(_cwd_now)

    st.markdown(f"""
    <style>
    .upload-wrapper {{
        max-width:620px; margin:40px auto 0;
        background:#f8faff; border-radius:24px;
        padding:40px 36px 32px; text-align:center;
        border:2px dashed #c4b5fd;
        box-shadow:0 8px 32px rgba(102,126,234,0.12);
    }}
    .upload-wrapper h2 {{ color:#1a1a2e; margin:12px 0 8px; font-size:1.7rem; }}
    .upload-wrapper p  {{ color:#6b7280; margin-bottom:0; font-size:0.97rem; line-height:1.6; }}
    .dbg-box {{
        max-width:620px; margin:12px auto 0;
        background:#fff8e1; border-radius:12px;
        padding:14px 20px; border:1px solid #ffd54f;
        font-size:0.82rem; color:#5d4037; text-align:left;
        word-break:break-all;
    }}
    </style>
    <div class="upload-wrapper">
        <div style="font-size:3rem;">📂</div>
        <h2>Upload Your Dataset</h2>
        <p>
            The file <b>ai_in_job_treands_without_skill.csv</b> was not found.<br>
            Either upload it below <b>OR</b> place it in the folder shown below and restart.
        </p>
    </div>
    <div class="dbg-box">
        <b>📍 App is running from:</b> {_cwd_now}<br>
        <b>📄 Files found there:</b> {', '.join(_files_now) if _files_now else 'none'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='max-width:620px;margin:16px auto 0;'>", unsafe_allow_html=True)
    _uploaded = st.file_uploader(
        "📎  Click to upload  ai_in_job_treands_without_skill.csv",
        type=["csv"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if _uploaded is None:
        st.stop()

    df = load_data(_uploaded)

dt_model, lr_model, km_model, le_dict, cls_features, reg_features, km_features = train_models(df)

INDUSTRIES    = sorted(df['Industry'].unique().tolist())
AI_IMPACTS    = ['Low', 'Moderate', 'High']
EDUCATIONS    = ['High School', 'Associate Degree', 'Bachelors Degree', 'Masters Degree', 'PhD']
LOCATIONS     = sorted(df['Location'].unique().tolist())

COLOR_SEQ     = ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#11998e','#f7971e','#38ef7d']
CLUSTER_COLORS = ['#667eea','#f5576c','#11998e','#f7971e']
CLUSTER_NAMES  = ['💼 High Earners – Safe Jobs',
                  '🚀 Rising Stars – Mid Range',
                  '⚠️ At Risk – Entry Level',
                  '🏆 Elite Professionals']


# ─── HELPER: encode single row ───────────────────────────────────────────────
def encode_row(row_dict, cols):
    encoded = {}
    for col in cols:
        val = row_dict[col]
        if col in le_dict:
            encoded[col] = le_dict[col].transform([val])[0]
        else:
            encoded[col] = val
    return pd.DataFrame([encoded])


# ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div style='font-size:2.8rem;'>🤖</div>
        <h2>AI Job Trends</h2>
        <p>Smart Career Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "📊  Explore Jobs",
         "🔮  Job Growth",
         "💰  Salary Finder",
         "🧩  Job Groups",
         "ℹ️   About"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:16px;background:rgba(255,255,255,0.05);border-radius:12px;text-align:center;'>
        <p style='color:rgba(255,255,255,0.5);font-size:0.78rem;margin:0;'>
        📁 30,000 Job Records<br>🌍 8 Countries · 8 Industries
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🤖 AI Job Trends Dashboard</div>
        <div class='hero-subtitle'>Discover how Artificial Intelligence is reshaping careers worldwide.<br>
        Explore trends, predict your job future, and find your earning potential.</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='icon'>📁</div>
            <div class='value'>30,000</div>
            <div class='label'>Jobs Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card metric-card-green'>
            <div class='icon'>🏭</div>
            <div class='value'>8</div>
            <div class='label'>Industries Covered</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card metric-card-blue'>
            <div class='icon'>🌍</div>
            <div class='value'>8</div>
            <div class='label'>Countries</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card metric-card-orange'>
            <div class='icon'>💰</div>
            <div class='value'>$70K</div>
            <div class='label'>Avg. Salary</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🚀 What Can You Do Here?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Pick a section below to get started</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);border-radius:20px;
                    padding:32px 24px;text-align:center;color:white;
                    box-shadow:0 8px 32px rgba(102,126,234,0.35);min-height:200px;'>
            <div style='font-size:3rem;'>📊</div>
            <div style='font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>Explore Jobs</div>
            <div style='opacity:0.85;font-size:0.95rem;'>
                See charts and trends about jobs, salaries, and industries around the world.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#11998e,#38ef7d);border-radius:20px;
                    padding:32px 24px;text-align:center;color:white;
                    box-shadow:0 8px 32px rgba(17,153,142,0.35);min-height:200px;'>
            <div style='font-size:3rem;'>🔮</div>
            <div style='font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>Will My Job Grow?</div>
            <div style='opacity:0.85;font-size:0.95rem;'>
                Answer a few questions and find out if your job is growing or shrinking in the AI era.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f093fb,#f5576c);border-radius:20px;
                    padding:32px 24px;text-align:center;color:white;
                    box-shadow:0 8px 32px rgba(245,87,108,0.35);min-height:200px;'>
            <div style='font-size:3rem;'>💰</div>
            <div style='font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>Salary Finder</div>
            <div style='opacity:0.85;font-size:0.95rem;'>
                Discover how much you could earn based on your skills, experience, and location.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#4facfe,#00f2fe);border-radius:20px;
                    padding:32px 24px;text-align:center;color:white;
                    box-shadow:0 8px 32px rgba(79,172,254,0.35);min-height:180px;'>
            <div style='font-size:3rem;'>🧩</div>
            <div style='font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>Job Groups</div>
            <div style='opacity:0.85;font-size:0.95rem;'>
                See which group of workers you belong to and what that means for your future.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f7971e,#ffd200);border-radius:20px;
                    padding:32px 24px;text-align:center;color:#1a1a2e;
                    box-shadow:0 8px 32px rgba(247,151,30,0.35);min-height:180px;'>
            <div style='font-size:3rem;'>💡</div>
            <div style='font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>Smart Insights</div>
            <div style='opacity:0.8;font-size:0.95rem;'>
                This dashboard analyzes real data from 30,000 job records across the globe to give you accurate insights.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;color:#9ca3af;font-size:0.85rem;padding:16px;'>
        📌 Use the sidebar on the left to navigate between sections
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE JOBS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Explore Jobs":
    st.markdown("""
    <div class='page-title-strip'>
        <div class='pt-icon'>📊</div>
        <div class='pt-text'>
            <h2>Explore Jobs</h2>
            <p>Filter, explore and understand job trends, salaries and AI impact worldwide</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FILTER BAR ────────────────────────────────────────────────────────────
    with st.expander("🔽  Filter the Data", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_industries = ['All Industries'] + sorted(df['Industry'].unique().tolist())
            sel_industry = st.selectbox('🏭 Industry', all_industries)
        with fc2:
            all_countries = ['All Countries'] + sorted(df['Location'].unique().tolist())
            sel_country = st.selectbox('🌍 Country', all_countries)
        with fc3:
            all_edu = ['All Education Levels'] + ['High School','Associate Degree','Bachelors Degree','Masters Degree','PhD']
            sel_edu = st.selectbox('🎓 Education', all_edu)

    # Apply filters
    dff = df.copy()
    if sel_industry != 'All Industries':
        dff = dff[dff['Industry'] == sel_industry]
    if sel_country != 'All Countries':
        dff = dff[dff['Location'] == sel_country]
    if sel_edu != 'All Education Levels':
        dff = dff[dff['Required_Education'] == sel_edu]

    st.markdown(f"""
    <div style='text-align:right;color:#9ca3af;font-size:0.85rem;margin-bottom:16px;'>
        Showing <strong style='color:#667eea;'>{len(dff):,}</strong> records
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    avg_sal  = dff['Median_Salary_USD'].mean() if len(dff) > 0 else 0
    grow_pct = (dff['Job_Status'] == 'Increasing').mean() * 100 if len(dff) > 0 else 0
    avg_risk = dff['Automation_Risk_'].mean() if len(dff) > 0 else 0
    avg_exp  = dff['Experience_Required_Years'].mean() if len(dff) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='icon'>💵</div>
            <div class='value'>${avg_sal:,.0f}</div>
            <div class='label'>Average Salary</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card metric-card-green'>
            <div class='icon'>📈</div>
            <div class='value'>{grow_pct:.1f}%</div>
            <div class='label'>Jobs Growing</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card metric-card-orange'>
            <div class='icon'>🤖</div>
            <div class='value'>{avg_risk:.1f}%</div>
            <div class='label'>Avg AI Risk</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card metric-card-blue'>
            <div class='icon'>⏳</div>
            <div class='value'>{avg_exp:.1f} yrs</div>
            <div class='label'>Avg Experience</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    if len(dff) == 0:
        st.warning("No records match the selected filters. Please adjust your filters.")
    else:
        # ── ROW 1: Job Status Donut + AI Impact Grouped Bar ───────────────────
        col1, col2 = st.columns(2)
        with col1:
            status_counts = dff['Job_Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            status_counts['Pct'] = (status_counts['Count'] / status_counts['Count'].sum() * 100).round(1)
            fig_donut = px.pie(
                status_counts, values='Count', names='Status',
                title='Are Jobs Growing or Shrinking?',
                color='Status',
                color_discrete_map={'Increasing': '#38ef7d', 'Decreasing': '#f5576c'},
                hole=0.55
            )
            fig_donut.update_traces(
                textinfo='percent+label',
                textfont_size=13,
                pull=[0.04, 0]
            )
            fig_donut.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                title_font=dict(size=17, color='#1a1a2e', family='Inter'),
                legend=dict(font=dict(size=13), orientation='h', yanchor='bottom', y=-0.15),
                margin=dict(t=50, b=40, l=20, r=20),
                height=280,
                annotations=[dict(
                    text=f"<b>{len(dff):,}</b><br>Jobs",
                    x=0.5, y=0.5, font_size=14, showarrow=False,
                    font_color='#1a1a2e'
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            ind_status = dff.groupby(['Industry','Job_Status']).size().reset_index(name='Count')
            fig_grp = px.bar(
                ind_status, x='Industry', y='Count', color='Job_Status',
                title='Job Growth vs Decline by Industry',
                color_discrete_map={'Increasing':'#38ef7d','Decreasing':'#f5576c'},
                barmode='group',
                text='Count'
            )
            fig_grp.update_traces(textposition='outside', textfont_size=10)
            fig_grp.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                title_font=dict(size=17, color='#1a1a2e', family='Inter'),
                xaxis_title='', yaxis_title='Number of Jobs',
                legend=dict(title='', font=dict(size=12)),
                margin=dict(t=50, b=20, l=20, r=20),
                height=280
            )
            st.plotly_chart(fig_grp, use_container_width=True)

        # ── ROW 2: Education vs Salary + Country Salary ───────────────────────
        col3, col4 = st.columns(2)
        with col3:
            edu_order = ['High School','Associate Degree','Bachelors Degree','Masters Degree','PhD']
            edu_sal = dff.groupby('Required_Education')['Median_Salary_USD'].mean().reindex(edu_order).reset_index()
            edu_sal.columns = ['Education','Avg_Salary']
            fig_edu = px.bar(
                edu_sal, x='Education', y='Avg_Salary',
                title='How Education Affects Your Salary',
                color='Avg_Salary',
                color_continuous_scale='Purples',
                text=edu_sal['Avg_Salary'].apply(lambda x: f'${x:,.0f}' if not pd.isna(x) else '')
            )
            fig_edu.update_traces(textposition='outside', textfont_size=11)
            fig_edu.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                title_font=dict(size=17, color='#1a1a2e', family='Inter'),
                showlegend=False, coloraxis_showscale=False,
                xaxis_title='', yaxis_title='Average Salary (USD)',
                xaxis_tickangle=-15,
                margin=dict(t=50, b=60, l=20, r=20), height=280
            )
            st.plotly_chart(fig_edu, use_container_width=True)

        with col4:
            loc_sal = dff.groupby('Location')['Median_Salary_USD'].mean().sort_values(ascending=True).reset_index()
            fig_loc = px.bar(
                loc_sal, x='Median_Salary_USD', y='Location',
                orientation='h',
                title='Average Salary by Country',
                color='Median_Salary_USD',
                color_continuous_scale='Blues',
                text=loc_sal['Median_Salary_USD'].apply(lambda x: f'${x:,.0f}')
            )
            fig_loc.update_traces(textposition='outside', textfont_size=11)
            fig_loc.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                title_font=dict(size=17, color='#1a1a2e', family='Inter'),
                showlegend=False, coloraxis_showscale=False,
                xaxis_title='Average Salary (USD)', yaxis_title='',
                margin=dict(t=50, b=20, l=20, r=80), height=280
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        # ── ROW 3: AI Risk by Industry (full width) ───────────────────────────
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        risk_ind = dff.groupby('Industry')['Automation_Risk_'].mean().sort_values(ascending=True).reset_index()
        fig_risk = px.bar(
            risk_ind, x='Automation_Risk_', y='Industry',
            orientation='h',
            title='Which Industries Face the Most Risk of AI Replacing Jobs?',
            color='Automation_Risk_',
            color_continuous_scale='RdYlGn_r',
            text=risk_ind['Automation_Risk_'].apply(lambda x: f'{x:.1f}%')
        )
        fig_risk.update_traces(textposition='outside', textfont_size=12)
        fig_risk.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=17, color='#1a1a2e', family='Inter'),
            showlegend=False, coloraxis_showscale=False,
            xaxis_title='Average AI Risk (%)', yaxis_title='',
            margin=dict(t=50, b=20, l=20, r=80), height=310
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── ROW 4: Experience vs Salary Scatter ───────────────────────────────
        scat_df = dff.sample(min(2000, len(dff)), random_state=42)
        fig_scat = px.scatter(
            scat_df,
            x='Experience_Required_Years', y='Median_Salary_USD',
            color='Job_Status',
            color_discrete_map={'Increasing':'#38ef7d','Decreasing':'#f5576c'},
            title='Does More Experience Mean Higher Salary?',
            labels={
                'Experience_Required_Years': 'Years of Experience',
                'Median_Salary_USD': 'Salary (USD)',
                'Job_Status': 'Job Trend'
            },
            opacity=0.6
        )
        _x = scat_df['Experience_Required_Years'].values
        _y = scat_df['Median_Salary_USD'].values
        _mask = ~(np.isnan(_x) | np.isnan(_y))
        if _mask.sum() > 2:
            _m, _b = np.polyfit(_x[_mask], _y[_mask], 1)
            _xr = np.linspace(_x[_mask].min(), _x[_mask].max(), 100)
            fig_scat.add_trace(go.Scatter(
                x=_xr, y=_m*_xr+_b,
                mode='lines', name='Trend',
                line=dict(color='#667eea', width=2, dash='dash')
            ))
        fig_scat.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=17, color='#1a1a2e', family='Inter'),
            legend=dict(title='', font=dict(size=12)),
            margin=dict(t=50, b=20, l=20, r=20), height=330
        )
        st.plotly_chart(fig_scat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WILL MY JOB GROW?
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Job Growth":
    st.markdown("""
    <div class='page-title-strip'>
        <div class='pt-icon'>🔮</div>
        <div class='pt-text'>
            <h2>Will My Job Grow? &nbsp;<span style='font-size:0.85rem;font-weight:500;
                background:rgba(255,255,255,0.15);border-radius:20px;padding:4px 14px;
                letter-spacing:1px;vertical-align:middle;'></span></h2>
            <p>Fill in the details below and find out if your job is on the rise or at risk</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            industry     = st.selectbox("🏭 Your Industry", INDUSTRIES)
            ai_impact    = st.selectbox("🤖 How Much AI Affects Your Job", AI_IMPACTS)
            education    = st.selectbox("🎓 Your Education Level", EDUCATIONS)
            location     = st.selectbox("🌍 Your Country", LOCATIONS)
        with col2:
            salary       = st.slider("💵 Your Current Salary (USD)", 28000, 145000, 70000, step=1000,
                                     format="$%d")
            experience   = st.slider("⏳ Years of Experience", 0, 20, 5)
            remote_ratio = st.slider("🏠 How Much You Work From Home (%)", 0, 100, 50)
        with col3:
            auto_risk    = st.slider("⚠️ Risk of AI Replacing Your Job (%)", 0, 100, 50)
            gender_div   = st.slider("👥 Team Gender Diversity (%)", 20, 80, 50)
            job_open_24  = st.number_input("📋 Job Openings Available Now", 100, 10000, 5000, step=100)
            proj_open_30 = st.number_input("📅 Expected Job Openings by 2030", 100, 10000, 5000, step=100)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Check My Job Status", key="cls_btn"):
        row = {
            'Industry': industry, 'AI_Impact_Level': ai_impact,
            'Median_Salary_USD': salary, 'Experience_Required_Years': experience,
            'Job_Openings_2024': job_open_24, 'Projected_Openings_2030': proj_open_30,
            'Remote_Work_Ratio_': remote_ratio, 'Automation_Risk_': auto_risk,
            'Location': location, 'Required_Education': education,
            'Gender_Diversity_': gender_div
        }
        X_pred = encode_row(row, cls_features)
        pred   = dt_model.predict(X_pred)[0]
        label  = le_dict['Job_Status'].inverse_transform([pred])[0]

        proba  = dt_model.predict_proba(X_pred)[0]
        conf   = proba.max() * 100

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col_r, col_s = st.columns([1, 1])

        with col_r:
            if label == 'Increasing':
                st.markdown(f"""
                <div class='result-growing'>
                    <div class='result-title'>Your Job Outlook</div>
                    <div class='result-value'>✅ Growing!</div>
                    <div class='result-desc'>
                        Great news! Jobs like yours are <strong>on the rise</strong>.<br>
                        The demand for your skills is increasing in the {industry} industry.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-shrinking'>
                    <div class='result-title'>Your Job Outlook</div>
                    <div class='result-value'>⚠️ Shrinking</div>
                    <div class='result-desc'>
                        Heads up! Jobs like yours may be <strong>declining</strong>.<br>
                        Consider upskilling or exploring adjacent roles in {industry}.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_s:
            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={'text': "How Sure Are We?", 'font': {'size': 16, 'color': '#1a1a2e'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': '#667eea'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#667eea'},
                    'steps': [
                        {'range': [0, 50], 'color': '#f3f4f6'},
                        {'range': [50, 75], 'color': '#ddd6fe'},
                        {'range': [75, 100], 'color': '#c4b5fd'}
                    ],
                    'threshold': {'line': {'color': '#764ba2', 'width': 4}, 'value': conf}
                }
            ))
            fig_gauge.update_layout(
                height=260, margin=dict(t=40, b=0, l=30, r=30),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Key factors
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header' style='font-size:1.3rem;'>🔑 What's Influencing This Result?</div>", unsafe_allow_html=True)

        factors = {
            '⏳ Experience':    experience / 20 * 100,
            '⚠️ AI Risk':       100 - auto_risk,
            '🏠 Remote Work':   remote_ratio,
            '💵 Salary Level':  (salary - 28000) / (145000 - 28000) * 100,
            '🤖 AI Impact':     {'Low': 80, 'Moderate': 50, 'High': 20}[ai_impact]
        }
        for name, val in factors.items():
            col_n, col_b = st.columns([1, 4])
            with col_n:
                st.markdown(f"<p style='margin:6px 0;font-weight:600;color:#374151;'>{name}</p>", unsafe_allow_html=True)
            with col_b:
                color = '#38ef7d' if val >= 60 else '#f7971e' if val >= 40 else '#f5576c'
                st.markdown(f"""
                <div style='background:#f3f4f6;border-radius:8px;height:28px;margin:4px 0;overflow:hidden;'>
                    <div style='width:{val:.0f}%;height:100%;background:{color};
                                border-radius:8px;display:flex;align-items:center;
                                padding-left:10px;color:white;font-weight:600;font-size:0.85rem;'>
                        {val:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SALARY FINDER
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰  Salary Finder":
    st.markdown("""
    <div class='page-title-strip'>
        <div class='pt-icon'>💰</div>
        <div class='pt-text'>
            <h2>Salary Finder &nbsp;<span style='font-size:0.85rem;font-weight:500;
                background:rgba(255,255,255,0.15);border-radius:20px;padding:4px 14px;
                letter-spacing:1px;vertical-align:middle;'></span></h2>
            <p>Tell us about yourself and we'll estimate how much you could earn</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            r_industry   = st.selectbox("🏭 Your Industry", INDUSTRIES, key='r_ind')
            r_ai_impact  = st.selectbox("🤖 How Much AI Affects Your Job", AI_IMPACTS, key='r_ai')
            r_education  = st.selectbox("🎓 Your Education Level", EDUCATIONS, key='r_edu')
            r_location   = st.selectbox("🌍 Your Country", LOCATIONS, key='r_loc')
        with col2:
            r_experience  = st.slider("⏳ Years of Experience", 0, 20, 5, key='r_exp')
            r_remote      = st.slider("🏠 How Much You Work From Home (%)", 0, 100, 50, key='r_rem')
            r_auto_risk   = st.slider("⚠️ Risk of AI Replacing Your Job (%)", 0, 100, 50, key='r_aut')
        with col3:
            r_gender_div  = st.slider("👥 Team Gender Diversity (%)", 20, 80, 50, key='r_gen')
            r_job_open_24 = st.number_input("📋 Job Openings Available Now", 100, 10000, 5000, step=100, key='r_jo')
            r_proj_30     = st.number_input("📅 Expected Job Openings by 2030", 100, 10000, 5000, step=100, key='r_pr')

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("💰  Find My Salary", key="reg_btn"):
        row = {
            'Industry': r_industry, 'AI_Impact_Level': r_ai_impact,
            'Experience_Required_Years': r_experience,
            'Job_Openings_2024': r_job_open_24, 'Projected_Openings_2030': r_proj_30,
            'Remote_Work_Ratio_': r_remote, 'Automation_Risk_': r_auto_risk,
            'Location': r_location, 'Required_Education': r_education,
            'Gender_Diversity_': r_gender_div
        }
        X_r    = encode_row(row, reg_features)
        pred_s = lr_model.predict(X_r)[0]
        pred_s = max(28000, min(145000, pred_s))
        low_s  = pred_s * 0.88
        high_s = pred_s * 1.12

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col_main, col_side = st.columns([1, 1])

        with col_main:
            st.markdown(f"""
            <div class='result-salary'>
                <div class='result-title'>Your Estimated Annual Salary</div>
                <div class='result-value'>💵 ${pred_s:,.0f}</div>
                <div class='result-desc'>
                    Salary Range: <strong>${low_s:,.0f}</strong> – <strong>${high_s:,.0f}</strong> per year<br>
                    Based on your experience, education, and industry in {r_location}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_side:
            # Salary band chart
            categories = ['Lower End', 'Your Estimate', 'Higher End']
            values     = [low_s, pred_s, high_s]
            colors     = ['#c4b5fd', '#667eea', '#764ba2']
            fig_bar = go.Figure(go.Bar(
                x=categories, y=values,
                marker_color=colors,
                text=[f'${v:,.0f}' for v in values],
                textposition='outside', textfont=dict(size=13, color='#1a1a2e')
            ))
            fig_bar.update_layout(
                title='Your Salary Range',
                title_font=dict(size=16, color='#1a1a2e', family='Inter'),
                plot_bgcolor='white', paper_bgcolor='white',
                yaxis=dict(tickprefix='$', showgrid=True, gridcolor='#f3f4f6'),
                xaxis_title='', yaxis_title='',
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20),
                height=280
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # What affects salary
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header' style='font-size:1.3rem;'>💡 What's Driving Your Salary?</div>", unsafe_allow_html=True)

        drivers = {
            '⏳ Years of Experience':    r_experience / 20 * 100,
            '⚠️ AI Replacement Risk (lower = better)': 100 - r_auto_risk,
            '🏠 Remote Work Flexibility': r_remote,
            '🎓 Education Level':         {'High School': 20, 'Associate Degree': 40,
                                           'Bachelors Degree': 60, 'Masters Degree': 80, 'PhD': 100}[r_education],
            '📋 Job Demand (Openings)':  min(r_job_open_24 / 10000 * 100, 100)
        }
        for name, val in drivers.items():
            col_n, col_b = st.columns([1.5, 4])
            with col_n:
                st.markdown(f"<p style='margin:6px 0;font-weight:600;color:#374151;'>{name}</p>", unsafe_allow_html=True)
            with col_b:
                color = '#667eea' if val >= 60 else '#f7971e' if val >= 35 else '#f5576c'
                st.markdown(f"""
                <div style='background:#f3f4f6;border-radius:8px;height:28px;margin:4px 0;overflow:hidden;'>
                    <div style='width:{val:.0f}%;height:100%;background:{color};
                                border-radius:8px;display:flex;align-items:center;
                                padding-left:10px;color:white;font-weight:600;font-size:0.85rem;'>
                        {val:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Industry comparison
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        ind_avg = df.groupby('Industry')['Median_Salary_USD'].mean().reset_index()
        ind_avg['highlight'] = ind_avg['Industry'].apply(
            lambda x: '🎯 Your Industry' if x == r_industry else 'Others'
        )
        fig_comp = px.bar(
            ind_avg.sort_values('Median_Salary_USD', ascending=True),
            x='Median_Salary_USD', y='Industry', orientation='h',
            title=f'How {r_industry} Compares to Other Industries',
            color='highlight',
            color_discrete_map={'🎯 Your Industry': '#667eea', 'Others': '#e5e7eb'},
            text=ind_avg.sort_values('Median_Salary_USD', ascending=True)['Median_Salary_USD'].apply(lambda x: f'${x:,.0f}')
        )
        fig_comp.update_traces(textposition='outside', textfont_size=11)
        fig_comp.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            showlegend=True, xaxis_title='Average Salary (USD)', yaxis_title='',
            margin=dict(t=50, b=20, l=20, r=80)
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — JOB GROUPS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧩  Job Groups":
    st.markdown("""
    <div class='page-title-strip'>
        <div class='pt-icon'>🧩</div>
        <div class='pt-text'>
            <h2>Job Groups &nbsp;<span style='font-size:0.85rem;font-weight:500;
                background:rgba(255,255,255,0.15);border-radius:20px;padding:4px 14px;
                letter-spacing:1px;vertical-align:middle;'></span></h2>
            <p>Discover which group of workers you belong to and what it means for your future</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EXPLANATION CARD ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#f0f4ff,#faf0ff);border-radius:18px;
                padding:24px 28px;border:1px solid #dde4ff;margin-bottom:24px;'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>
            <div style='font-size:1.8rem;'>🔍</div>
            <div style='font-weight:700;color:#1a1a2e;font-size:1.1rem;'>How Does This Work?</div>
        </div>
        <p style='color:#4b5563;margin:0;line-height:1.7;font-size:0.97rem;'>
            We looked at <strong>30,000 jobs</strong> and automatically grouped them into similar groups
            based on 4 things: <strong>💰 Salary</strong>, <strong>⚠️ AI Risk</strong>,
            <strong>⏳ Years of Experience</strong>, and <strong>🏠 Remote Work</strong>.
            Jobs that are similar to each other end up in the same group.
            Use the slider below to choose how many groups you want to see.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── CLUSTER SLIDER ────────────────────────────────────────────────────────
    n_clusters = st.slider("👥 How many groups would you like to see?", 2, 6, 4)

    # ── RUN KMEANS ────────────────────────────────────────────────────────────
    df_enc2 = df.copy()
    for col in ['Industry','AI_Impact_Level','Required_Education','Location','Job_Status']:
        df_enc2[col] = le_dict[col].transform(df[col])

    km_new   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = km_new.fit_predict(df_enc2[km_features])

    cluster_palette = ['#667eea','#f5576c','#11998e','#f7971e','#4facfe','#764ba2'][:n_clusters]

    cluster_stats = df.groupby('Cluster').agg(
        Avg_Salary=('Median_Salary_USD','mean'),
        Avg_Risk=('Automation_Risk_','mean'),
        Avg_Exp=('Experience_Required_Years','mean'),
        Avg_Remote=('Remote_Work_Ratio_','mean'),
        Count=('Median_Salary_USD','count')
    ).reset_index()

    cluster_stats_sorted = cluster_stats.sort_values('Avg_Salary', ascending=False).reset_index(drop=True)
    cluster_name_pool = [
        '🏆 Elite Professionals',
        '💼 High Earners',
        '🚀 Rising Stars',
        '🛡️ Stable Mid-Range',
        '⚠️ At Risk',
        '🔄 Entry Level'
    ]
    cluster_labels_map = {}
    for i, row in cluster_stats_sorted.iterrows():
        cluster_labels_map[int(row['Cluster'])] = cluster_name_pool[i % len(cluster_name_pool)]

    df['Group Name'] = df['Cluster'].map(cluster_labels_map)

    # ── INSIGHT CARDS (top) ───────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📋 Group Summaries</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Each group represents workers with similar salary, AI risk, experience and remote work</div>", unsafe_allow_html=True)

    n_cols = min(n_clusters, 4)
    card_rows = [cluster_stats_sorted.iloc[i:i+n_cols] for i in range(0, len(cluster_stats_sorted), n_cols)]
    for card_row in card_rows:
        cols = st.columns(len(card_row))
        for j, (_, stat) in enumerate(card_row.iterrows()):
            clr  = cluster_palette[int(stat['Cluster']) % len(cluster_palette)]
            name = cluster_labels_map[int(stat['Cluster'])]
            with cols[j]:
                st.markdown(f"""
                <div style='background:{clr};border-radius:18px;padding:22px 18px;color:white;
                            box-shadow:0 6px 24px rgba(0,0,0,0.15);margin-bottom:16px;'>
                    <div style='font-size:1.15rem;font-weight:700;margin-bottom:14px;
                                border-bottom:1px solid rgba(255,255,255,0.25);padding-bottom:10px;'>
                        {name}
                    </div>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                        <div style='background:rgba(255,255,255,0.2);border-radius:10px;
                                    padding:10px;text-align:center;'>
                            <div style='font-size:1.25rem;font-weight:800;'>${stat['Avg_Salary']:,.0f}</div>
                            <div style='font-size:0.75rem;opacity:0.9;margin-top:2px;'>Avg Salary</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.2);border-radius:10px;
                                    padding:10px;text-align:center;'>
                            <div style='font-size:1.25rem;font-weight:800;'>{stat['Avg_Risk']:.0f}%</div>
                            <div style='font-size:0.75rem;opacity:0.9;margin-top:2px;'>AI Risk</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.2);border-radius:10px;
                                    padding:10px;text-align:center;'>
                            <div style='font-size:1.25rem;font-weight:800;'>{stat['Avg_Exp']:.0f} yrs</div>
                            <div style='font-size:0.75rem;opacity:0.9;margin-top:2px;'>Experience</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.2);border-radius:10px;
                                    padding:10px;text-align:center;'>
                            <div style='font-size:1.25rem;font-weight:800;'>{stat['Avg_Remote']:.0f}%</div>
                            <div style='font-size:0.75rem;opacity:0.9;margin-top:2px;'>Remote Work</div>
                        </div>
                    </div>
                    <div style='margin-top:10px;font-size:0.82rem;opacity:0.85;text-align:center;
                                background:rgba(0,0,0,0.12);border-radius:8px;padding:6px;'>
                        {int(stat["Count"]):,} jobs in this group
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── CHARTS ROW 1: Scatter + Salary Bar ────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_sc = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x='Median_Salary_USD', y='Automation_Risk_',
            color='Group Name',
            color_discrete_sequence=cluster_palette,
            title='Jobs Grouped by Salary and AI Risk',
            labels={
                'Median_Salary_USD': 'Salary (USD)',
                'Automation_Risk_': 'Risk of AI Replacing Job (%)',
                'Group Name': 'Group'
            },
            opacity=0.65
        )
        fig_sc.update_traces(marker=dict(size=5))
        fig_sc.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            legend=dict(title='', font=dict(size=11),
                        orientation='h', yanchor='bottom', y=-0.35),
            margin=dict(t=50, b=80, l=20, r=20), height=330
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        sal_bar = cluster_stats_sorted.copy()
        sal_bar['Group'] = sal_bar['Cluster'].map(cluster_labels_map)
        fig_sal = px.bar(
            sal_bar, x='Group', y='Avg_Salary',
            title='Average Salary Per Group',
            color='Group',
            color_discrete_sequence=cluster_palette,
            text=sal_bar['Avg_Salary'].apply(lambda x: f'${x:,.0f}')
        )
        fig_sal.update_traces(textposition='outside', textfont_size=11)
        fig_sal.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            showlegend=False,
            xaxis_title='', yaxis_title='Average Salary (USD)',
            xaxis_tickangle=-15,
            margin=dict(t=50, b=80, l=20, r=20), height=330
        )
        st.plotly_chart(fig_sal, use_container_width=True)

    # ── CHARTS ROW 2: AI Risk Bar + Experience Bar ────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        fig_risk_bar = px.bar(
            sal_bar, x='Group', y='Avg_Risk',
            title='AI Replacement Risk Per Group',
            color='Group',
            color_discrete_sequence=cluster_palette,
            text=sal_bar['Avg_Risk'].apply(lambda x: f'{x:.0f}%')
        )
        fig_risk_bar.update_traces(textposition='outside', textfont_size=11)
        fig_risk_bar.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            showlegend=False,
            xaxis_title='', yaxis_title='Average AI Risk (%)',
            xaxis_tickangle=-15,
            margin=dict(t=50, b=80, l=20, r=20), height=310
        )
        st.plotly_chart(fig_risk_bar, use_container_width=True)

    with col4:
        fig_exp_bar = px.bar(
            sal_bar, x='Group', y='Avg_Exp',
            title='Years of Experience Per Group',
            color='Group',
            color_discrete_sequence=cluster_palette,
            text=sal_bar['Avg_Exp'].apply(lambda x: f'{x:.0f} yrs')
        )
        fig_exp_bar.update_traces(textposition='outside', textfont_size=11)
        fig_exp_bar.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            showlegend=False,
            xaxis_title='', yaxis_title='Average Experience (Years)',
            xaxis_tickangle=-15,
            margin=dict(t=50, b=80, l=20, r=20), height=310
        )
        st.plotly_chart(fig_exp_bar, use_container_width=True)

    # ── KEY INSIGHTS SECTION ──────────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💡 Key Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>What the data is telling us</div>", unsafe_allow_html=True)

    # Compute dynamic insights
    top_grp   = cluster_stats_sorted.iloc[0]
    bot_grp   = cluster_stats_sorted.iloc[-1]
    top_name  = cluster_labels_map[int(top_grp['Cluster'])]
    bot_name  = cluster_labels_map[int(bot_grp['Cluster'])]
    at_risk_pct = int(bot_grp['Count'] / cluster_stats_sorted['Count'].sum() * 100)

    insights = [
        ("⏳", "Experience = Protection",
         f"Moving from the lowest group ({bot_grp['Avg_Exp']:.0f} yrs experience) to the top group "
         f"({top_grp['Avg_Exp']:.0f} yrs) drops AI risk from "
         f"{bot_grp['Avg_Risk']:.0f}% down to {top_grp['Avg_Risk']:.0f}%."),
        ("💰", "Higher Salary = Safer Job",
         f"{top_name} earns ${top_grp['Avg_Salary']:,.0f} on average with only {top_grp['Avg_Risk']:.0f}% AI risk, "
         f"while {bot_name} earns ${bot_grp['Avg_Salary']:,.0f} with {bot_grp['Avg_Risk']:.0f}% risk. "
         "AI replaces cheap, repetitive work first."),
        ("⚠️", f"{at_risk_pct}% of Jobs Are Highly Vulnerable",
         f"About {int(bot_grp['Count']):,} jobs fall into the most at-risk group. "
         "These are typically entry-level roles with lower pay and high automation potential."),
        ("🏠", "Remote Work Follows Seniority",
         f"The top group enjoys {top_grp['Avg_Remote']:.0f}% remote work flexibility, "
         f"while the lowest group sits at {bot_grp['Avg_Remote']:.0f}%. "
         "Better jobs come with more freedom to work from home."),
    ]

    for icon, title, desc in insights:
        st.markdown(f"""
        <div style='background:white;border-radius:14px;padding:20px 24px;
                    border:1px solid #e8edf5;box-shadow:0 2px 12px rgba(0,0,0,0.05);
                    margin-bottom:12px;display:flex;gap:16px;align-items:flex-start;'>
            <div style='font-size:2rem;flex-shrink:0;'>{icon}</div>
            <div>
                <div style='font-weight:700;color:#1a1a2e;font-size:1rem;margin-bottom:4px;'>{title}</div>
                <div style='color:#6b7280;font-size:0.93rem;line-height:1.6;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── USER PREDICTION SECTION ───────────────────────────────────────────────
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);border-radius:20px;
                padding:28px 32px;margin-bottom:24px;'>
        <div style='color:white;font-size:1.5rem;font-weight:700;margin-bottom:6px;'>
            🎯 Which Group Do You Belong To?
        </div>
        <div style='color:rgba(255,255,255,0.7);font-size:0.97rem;'>
            Fill in your details below and we'll tell you which job group you fall into
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        uc1, uc2 = st.columns(2)
        with uc1:
            u_salary   = st.slider("💵 Your Current or Expected Salary (USD)",
                                   28000, 145000, 65000, step=1000,
                                   format="$%d", key='cl_sal')
            u_risk     = st.slider("⚠️ Risk of AI Replacing Your Job (%)",
                                   0, 100, 50, key='cl_risk')
        with uc2:
            u_exp      = st.slider("⏳ Your Years of Experience",
                                   0, 20, 5, key='cl_exp')
            u_remote   = st.slider("🏠 How Much You Work From Home (%)",
                                   0, 100, 50, key='cl_remote')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Find My Job Group", key="cluster_predict_btn"):
        user_point = np.array([[u_salary, u_risk, u_exp, u_remote]])
        pred_cluster = km_new.predict(user_point)[0]
        pred_name    = cluster_labels_map.get(int(pred_cluster), "Unknown Group")
        pred_color   = cluster_palette[int(pred_cluster) % len(cluster_palette)]
        pred_stats   = cluster_stats[cluster_stats['Cluster'] == pred_cluster].iloc[0]

        st.markdown(f"""
        <div style='background:{pred_color};border-radius:20px;padding:32px;
                    color:white;box-shadow:0 12px 40px rgba(0,0,0,0.2);margin-bottom:20px;'>
            <div style='font-size:0.95rem;opacity:0.85;margin-bottom:4px;'>You Belong To</div>
            <div style='font-size:2.2rem;font-weight:800;margin-bottom:16px;'>{pred_name}</div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
                <div style='background:rgba(255,255,255,0.2);border-radius:12px;
                            padding:14px;text-align:center;'>
                    <div style='font-size:1.3rem;font-weight:800;'>${pred_stats['Avg_Salary']:,.0f}</div>
                    <div style='font-size:0.78rem;opacity:0.9;margin-top:3px;'>Group Avg Salary</div>
                </div>
                <div style='background:rgba(255,255,255,0.2);border-radius:12px;
                            padding:14px;text-align:center;'>
                    <div style='font-size:1.3rem;font-weight:800;'>{pred_stats['Avg_Risk']:.0f}%</div>
                    <div style='font-size:0.78rem;opacity:0.9;margin-top:3px;'>Group AI Risk</div>
                </div>
                <div style='background:rgba(255,255,255,0.2);border-radius:12px;
                            padding:14px;text-align:center;'>
                    <div style='font-size:1.3rem;font-weight:800;'>{pred_stats['Avg_Exp']:.0f} yrs</div>
                    <div style='font-size:0.78rem;opacity:0.9;margin-top:3px;'>Group Avg Exp</div>
                </div>
                <div style='background:rgba(255,255,255,0.2);border-radius:12px;
                            padding:14px;text-align:center;'>
                    <div style='font-size:1.3rem;font-weight:800;'>{int(pred_stats['Count']):,}</div>
                    <div style='font-size:0.78rem;opacity:0.9;margin-top:3px;'>Jobs in Group</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show user point on the scatter
        samp = df.sample(min(2000, len(df)), random_state=42)
        fig_user = px.scatter(
            samp,
            x='Median_Salary_USD', y='Automation_Risk_',
            color='Group Name',
            color_discrete_sequence=cluster_palette,
            title='Where Do You Sit Among All Jobs?',
            labels={
                'Median_Salary_USD': 'Salary (USD)',
                'Automation_Risk_': 'AI Risk (%)',
                'Group Name': 'Group'
            },
            opacity=0.45
        )
        fig_user.add_trace(go.Scatter(
            x=[u_salary], y=[u_risk],
            mode='markers+text',
            name='⭐ You',
            text=['  ⭐ You'],
            textposition='middle right',
            textfont=dict(size=14, color='#1a1a2e'),
            marker=dict(size=18, color='#ffd200',
                        line=dict(width=3, color='#1a1a2e'),
                        symbol='star')
        ))
        fig_user.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title_font=dict(size=16, color='#1a1a2e', family='Inter'),
            legend=dict(title='', font=dict(size=11),
                        orientation='h', yanchor='bottom', y=-0.35),
            margin=dict(t=50, b=100, l=20, r=20), height=350
        )
        st.plotly_chart(fig_user, use_container_width=True)

        # Advice card
        if 'Elite' in pred_name or 'High' in pred_name:
            advice_color = '#11998e'
            advice_icon  = '🌟'
            advice_text  = ("You're in a strong position! Your experience and salary level place you "
                            "among the most secure workers. Keep building specialised skills to stay ahead.")
        elif 'Rising' in pred_name or 'Mid' in pred_name or 'Stable' in pred_name:
            advice_color = '#f7971e'
            advice_icon  = '🚀'
            advice_text  = ("You're on a good path! Focus on gaining more experience and reducing your "
                            "exposure to routine tasks. Upskilling in AI-related areas can move you "
                            "into the top group.")
        else:
            advice_color = '#f5576c'
            advice_icon  = '💡'
            advice_text  = ("Your role has a higher risk of automation. Now is a great time to invest "
                            "in learning new skills, especially in areas that are harder for AI to "
                            "replicate, like leadership, creativity, or complex problem-solving.")

        st.markdown(f"""
        <div style='background:{advice_color};border-radius:16px;padding:22px 26px;
                    color:white;box-shadow:0 6px 24px rgba(0,0,0,0.15);margin-top:8px;
                    display:flex;gap:16px;align-items:flex-start;'>
            <div style='font-size:2.2rem;flex-shrink:0;'>{advice_icon}</div>
            <div>
                <div style='font-weight:700;font-size:1.05rem;margin-bottom:6px;'>
                    What This Means For You
                </div>
                <div style='opacity:0.92;line-height:1.6;font-size:0.95rem;'>{advice_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️   About":
    st.markdown("""
    <div class='page-title-strip'>
        <div class='pt-icon'>ℹ️</div>
        <div class='pt-text'>
            <h2>About This App</h2>
            <p>Everything you need to know about this dashboard</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='about-card'>
            <h3>🤖 What Is This App?</h3>
            <p style='color:#6b7280;line-height:1.7;'>
                This dashboard helps you understand how Artificial Intelligence
                is changing the job market. Whether you're curious about your
                career future, want to know what you could earn, or just want
                to explore job trends worldwide — this app has you covered.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='about-card'>
            <h3>📁 The Data</h3>
            <p style='color:#6b7280;line-height:1.7;'>
                This app is powered by a dataset of <strong>30,000 real-world job records</strong>
                spanning <strong>8 industries</strong> and <strong>8 countries</strong> including
                USA, UK, India, Germany, Canada, Australia, China, and Brazil.
                Each record contains information about salary, education, AI risk,
                remote work, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='about-card'>
            <h3>🔮 How Does the Prediction Work?</h3>
            <p style='color:#6b7280;line-height:1.7;'>
                Behind the scenes, this app uses smart computer programs trained
                on thousands of real job records to make predictions.<br><br>
                • <strong>Will My Job Grow?</strong> — learns patterns from job data to decide
                  if a job is growing or shrinking<br><br>
                • <strong>Salary Finder</strong> — calculates expected salary based on
                  your inputs like experience and education<br><br>
                • <strong>Job Groups</strong> — automatically groups similar jobs together
                  so you can see where you fit
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='about-card'>
            <h3>💡 Tips for Best Results</h3>
            <p style='color:#6b7280;line-height:1.7;'>
                • Be as accurate as possible when filling in your details<br>
                • Try different combinations to explore different outcomes<br>
                • Use the <strong>Explore Jobs</strong> page first to understand the trends<br>
                • The Salary Finder works best for full-time professional roles
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);border-radius:20px;
                padding:32px;text-align:center;margin-top:8px;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <div style='color:white;font-size:1.3rem;font-weight:700;margin:12px 0 8px;'>
            AI Job Trends Dashboard
        </div>
        <div style='color:rgba(255,255,255,0.65);font-size:0.95rem;'>
            Built with ❤️ using Python & Streamlit · Data: 30,000 Job Records · 8 Countries · 8 Industries
        </div>
    </div>
    """, unsafe_allow_html=True)

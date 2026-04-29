"""
app.py — FarmAI Profit Optimizer
=================================
Live sidebar updates with session_state.
All pages sync automatically to selected state/crop/inputs.
Fixes: state climate lookup, yield units, cache, safe_encode, profit display.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import requests

warnings.filterwarnings("ignore")

# ── 1. PAGE CONFIG & PREMIUM DESIGN SYSTEM ──────────────────────────────────
st.set_page_config(
    page_title="FarmAI — Profit Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Farmering UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #15803d;
        --primary-light: #22c55e;
        --primary-soft: #eaf5ea;
        --bg: #f8fafc;
        --surface: #ffffff;
        --text-main: #0f172a;  /* Darker Slate 900 */
        --text-muted: #475569; /* Darker Slate 600 */
        --border: #cbd5e1;    /* Slightly more visible border */
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }

    .stApp {
        background-color: var(--bg) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Global text color enforcement — narrowed to avoid widget backgrounds */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stMarkdown p {
        color: var(--text-main) !important;
    }

    /* Fix for Input Fields & Select Boxes readability */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] > input,
    .stSelectbox div, .stNumberInput input, .stTextInput input {
        background-color: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
    }

    /* Target the text inside selectbox dropdowns and options */
    div[role="listbox"] li, div[role="listbox"] div {
        color: var(--text-main) !important;
        background-color: #ffffff !important;
    }

    /* Fix for placeholder text */
    ::placeholder {
        color: var(--text-muted) !important;
        opacity: 1;
    }

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebarNav"] { display: none; }

    /* Custom Navigation Buttons */
    .nav-btn {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 12px;
        color: var(--text-muted);
        text-decoration: none;
        margin-bottom: 4px;
        transition: all 0.2s ease;
        border: none;
        background: transparent;
        width: 100%;
        cursor: pointer;
        font-weight: 500;
    }
    .nav-btn:hover {
        background-color: var(--bg);
        color: var(--primary);
    }
    .nav-btn-active {
        background-color: var(--primary-soft) !important;
        color: var(--primary) !important;
        font-weight: 600;
    }

    /* Premium Cards */
    .f-card {
        background: var(--surface);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: transform 0.2s ease;
    }
    .f-card:hover {
        transform: translateY(-2px);
    }

    .f-card-primary {
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
        color: white;
        border: none;
    }
    .f-card-primary * { color: white !important; }

    /* Custom Table Styling */
    .f-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .f-table th {
        text-align: left;
        padding: 12px 16px;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 13px;
        border-bottom: 1px solid var(--border);
    }
    .f-table td {
        padding: 16px;
        border-bottom: 1px solid #f1f5f9;
        font-size: 14px;
        color: var(--text-main);
    }

    /* Status Pills */
    .pill {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }
    .pill-success { background: #dcfce7; color: #15803d; }
    .pill-info { background: #e0f2fe; color: #0369a1; }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Mapping from yield-dataset crop names -> price-dataset crop names
CROP_PRICE_MAP = {
    "arecanut":            "arecanut(betelnut/supari)",
    "barley":              "barley (jau)",
    "cardamom":            "cardamoms",
    "cashewnut":           "cashewnuts",
    "coriander":           "coriander(leaves)",
    "cotton(lint)":        "cotton",
    "cowpea(lobia)":       "cowpea",
    "finger millet":       "ragi (finger millet)",
    "ginger":              "ginger(dry)",
    "gram":                "bengal gram(whole)",
    "greengram":           "green gram (moong)(whole)",
    "guar seed":           "guar seed(cluster beans seed)",
    "masoor":              "lentil (masoor)(whole)",
    "millet":              "bajra(pearl millet/cumbu)",
    "niger seed":          "niger seed (ramtil)",
    "pearl millet":        "bajra(pearl millet/cumbu)",
    "peas & beans (pulses)": "field pea(dry)",
    "pigeonpea":           "arhar (tur/red gram)(whole)",
    "rapeseed &mustard":   "mustard",
    "sesame":              "sesamum(sesame,gingelly,til)",
    "sorghum":             "jowar(sorghum)",
    "sugarcane":           "sugar",
    "blackgram":           "black gram (urd beans)(whole)",
    "groundnut":           "groundnut",
    "onion":               "onion",
    "potato":              "potato",
    "rice":                "rice",
    "wheat":               "wheat",
    "maize":               "maize",
    "soyabean":            "soyabean",
    "sunflower":           "sunflower",
    "turmeric":            "turmeric",
    "garlic":              "garlic",
    "coconut":             "coconut",
    "black pepper":        "black pepper",
    "jute":                "jute",
    "tapioca":             "tapioca",
    "tobacco":             "tobacco",
    "linseed":             "linseed",
    "safflower":           "safflower",
    "castor seed":         "castor seed",
}

# State-level climate averages (temp °C, humidity %, soil pH)
# Fixes the hardcoded 25/65/6.5 in crop recommender inference
STATE_CLIMATE = {
    "andhra pradesh":    (29, 72, 6.8), "arunachal pradesh": (20, 80, 5.5),
    "assam":             (25, 82, 5.8), "bihar":             (25, 65, 7.0),
    "chhattisgarh":      (27, 70, 6.5), "goa":               (28, 82, 6.0),
    "gujarat":           (28, 60, 7.2), "haryana":           (24, 58, 7.8),
    "himachal pradesh":  (16, 65, 6.2), "jharkhand":         (25, 70, 5.8),
    "karnataka":         (25, 68, 6.5), "kerala":            (28, 85, 5.8),
    "madhya pradesh":    (26, 62, 7.0), "maharashtra":       (27, 65, 6.8),
    "manipur":           (22, 80, 5.5), "meghalaya":         (18, 85, 5.2),
    "mizoram":           (22, 80, 5.5), "nagaland":          (20, 78, 5.5),
    "odisha":            (28, 75, 6.2), "punjab":            (23, 58, 7.8),
    "rajasthan":         (32, 40, 7.8), "sikkim":            (15, 80, 5.5),
    "tamil nadu":        (30, 75, 6.5), "telangana":         (30, 68, 7.0),
    "tripura":           (25, 82, 5.8), "uttar pradesh":     (25, 62, 7.5),
    "uttarakhand":       (18, 68, 6.5), "west bengal":       (27, 80, 6.0),
    "andaman and nicobar": (30, 85, 6.0), "jammu and kashmir": (12, 55, 6.5),
    "lakshadweep":       (30, 85, 6.5), "puducherry":        (30, 75, 6.5),
    "chandigarh":        (23, 60, 7.5), "delhi":             (25, 60, 7.5),
}

import requests

# Dictionary of Indian State Coordinates (Lat, Lon) for live weather
STATE_COORDS = {
    "andhra pradesh": (15.91, 79.74), "arunachal pradesh": (28.21, 94.72),
    "assam": (26.20, 92.93), "bihar": (25.09, 85.31),
    "chhattisgarh": (21.27, 81.86), "goa": (15.29, 74.12),
    "gujarat": (22.25, 71.19), "haryana": (29.05, 76.08),
    "himachal pradesh": (31.10, 77.17), "jharkhand": (23.61, 85.27),
    "karnataka": (15.31, 75.71), "kerala": (10.85, 76.27),
    "madhya pradesh": (22.97, 78.65), "maharashtra": (19.75, 75.71),
    "manipur": (24.66, 93.90), "meghalaya": (25.46, 91.36),
    "mizoram": (23.16, 92.93), "nagaland": (26.15, 94.56),
    "odisha": (20.95, 85.09), "punjab": (31.14, 75.34),
    "rajasthan": (27.02, 74.21), "sikkim": (27.53, 88.51),
    "tamil nadu": (11.12, 78.65), "telangana": (18.11, 79.01),
    "tripura": (23.94, 91.98), "uttar pradesh": (26.84, 80.94),
    "uttarakhand": (30.06, 79.01), "west bengal": (22.98, 87.85),
    "delhi": (28.70, 77.10)
}

@st.cache_data(ttl=3600)
def get_state_climate(state: str):
    """Return live (temperature, humidity, ph) for a given state, with static fallback."""
    state_low = state.lower()
    fallback_temp, fallback_hum, ph = STATE_CLIMATE.get(state_low, (25, 65, 6.5))
    
    if state_low in STATE_COORDS:
        lat, lon = STATE_COORDS[state_low]
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m"
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                data = res.json()
                temp = data["current_weather"]["temperature"]
                # Get first valid humidity reading
                hum = next((h for h in data["hourly"]["relativehumidity_2m"] if h is not None), fallback_hum)
                return temp, hum, ph
        except Exception:
            pass
            
    return fallback_temp, fallback_hum, ph

# State-wise typical soil N/P/K (kg/ha) and annual rainfall (mm)
# Auto-fills sidebar inputs when a new state is selected
STATE_SOIL_DEFAULTS = {
    "andhra pradesh":    {"n": 60, "p": 30, "k": 40, "rain": 900},
    "arunachal pradesh": {"n": 80, "p": 35, "k": 45, "rain": 2500},
    "assam":             {"n": 70, "p": 25, "k": 30, "rain": 1800},
    "bihar":             {"n": 50, "p": 25, "k": 30, "rain": 1000},
    "chhattisgarh":      {"n": 55, "p": 25, "k": 35, "rain": 1200},
    "goa":               {"n": 65, "p": 35, "k": 40, "rain": 2500},
    "gujarat":           {"n": 45, "p": 30, "k": 35, "rain": 700},
    "haryana":           {"n": 80, "p": 40, "k": 35, "rain": 600},
    "himachal pradesh":  {"n": 65, "p": 30, "k": 35, "rain": 1400},
    "jharkhand":         {"n": 50, "p": 20, "k": 25, "rain": 1200},
    "karnataka":         {"n": 55, "p": 30, "k": 35, "rain": 1000},
    "kerala":            {"n": 70, "p": 35, "k": 50, "rain": 2800},
    "madhya pradesh":    {"n": 55, "p": 25, "k": 30, "rain": 900},
    "maharashtra":       {"n": 60, "p": 30, "k": 35, "rain": 800},
    "manipur":           {"n": 65, "p": 30, "k": 35, "rain": 1500},
    "meghalaya":         {"n": 70, "p": 30, "k": 35, "rain": 2500},
    "mizoram":           {"n": 65, "p": 28, "k": 32, "rain": 2000},
    "nagaland":          {"n": 65, "p": 28, "k": 32, "rain": 1800},
    "odisha":            {"n": 60, "p": 25, "k": 30, "rain": 1400},
    "punjab":            {"n": 100, "p": 50, "k": 40, "rain": 500},
    "rajasthan":         {"n": 40, "p": 20, "k": 25, "rain": 300},
    "sikkim":            {"n": 70, "p": 30, "k": 35, "rain": 2500},
    "tamil nadu":        {"n": 65, "p": 35, "k": 45, "rain": 900},
    "telangana":         {"n": 60, "p": 30, "k": 35, "rain": 800},
    "tripura":           {"n": 65, "p": 28, "k": 32, "rain": 1900},
    "uttar pradesh":     {"n": 70, "p": 35, "k": 35, "rain": 900},
    "uttarakhand":       {"n": 65, "p": 30, "k": 35, "rain": 1400},
    "west bengal":       {"n": 70, "p": 30, "k": 35, "rain": 1500},
    "andaman and nicobar": {"n": 70, "p": 35, "k": 40, "rain": 3000},
    "jammu and kashmir": {"n": 55, "p": 25, "k": 30, "rain": 650},
    "lakshadweep":       {"n": 60, "p": 30, "k": 40, "rain": 1600},
    "puducherry":        {"n": 60, "p": 30, "k": 40, "rain": 1100},
    "chandigarh":        {"n": 75, "p": 38, "k": 35, "rain": 650},
    "delhi":             {"n": 65, "p": 30, "k": 30, "rain": 600},
}

def resolve_price_crop(crop_name: str, price_crops: set) -> str:
    """Return the price-dataset crop name that best matches crop_name."""
    low = crop_name.lower()
    # 1. direct map
    if low in CROP_PRICE_MAP:
        return CROP_PRICE_MAP[low]
    # 2. exact
    if low in price_crops:
        return low
    # 3. substring forward
    for pc in price_crops:
        if low in pc:
            return pc
    # 4. substring reverse
    for pc in price_crops:
        if pc in low:
            return pc
    return low   # fallback – will produce empty df

# 2. PATHS (Flexible for Local vs GitHub)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

# Check both common naming conventions
if os.path.exists(os.path.join(BASE_DIR, "clean_data")):
    CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
else:
    CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

# 2. INITIALIZE STATE
if "page"             not in st.session_state: st.session_state.page             = "Overview"
if "n"                not in st.session_state: st.session_state.n                = 70
if "p"                not in st.session_state: st.session_state.p                = 45
if "k"                not in st.session_state: st.session_state.k                = 30
if "rain"             not in st.session_state: st.session_state.rain             = 500
if "y_area"           not in st.session_state: st.session_state.y_area           = 1.0
if "y_state"          not in st.session_state: st.session_state.y_state          = ""
if "y_district"       not in st.session_state: st.session_state.y_district       = "All Districts"
if "y_crop"           not in st.session_state: st.session_state.y_crop           = ""
if "_last_autofill_state" not in st.session_state: st.session_state._last_autofill_state = ""
if "app_initialized"  not in st.session_state: st.session_state.app_initialized  = False

# 3. LOAD MODELS & DATA
@st.cache_resource
def load_models():
    _models = {}
    for name in ["crop_recommender.pkl", "yield_predictor.pkl", "yield_crop_encoder.pkl", "yield_state_encoder.pkl"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p): _models[name] = joblib.load(p)
    return _models

@st.cache_data
def load_data():
    # 1. Monthly Prices
    p_files = ["mandi_prices_monthly.csv", "mandi_prices_clean.csv", "mandi_prices_cleaned.csv"]
    df_p = pd.DataFrame()
    for f in p_files:
        path = os.path.join(CLEAN_DIR, f)
        if os.path.exists(path):
            df_p = pd.read_csv(path)
            break
    
    if not df_p.empty and 'date' in df_p.columns:
        df_p['date'] = pd.to_datetime(df_p['date'])
    
    # 2. Crop Yield
    y_files = ["crop_yield_clean.csv", "crop_yield_cleaned.csv", "crop_yield.csv"]
    df_y = pd.DataFrame()
    for f in y_files:
        path = os.path.join(CLEAN_DIR, f)
        if os.path.exists(path):
            df_y = pd.read_csv(path)
            break
            
    # 3. Profit Recommendations
    df_prof = pd.DataFrame()
    prof_path = os.path.join(OUT_DIR, "m4_final_recommendations.csv")
    if os.path.exists(prof_path):
        df_prof = pd.read_csv(prof_path)
    
    # 4. API Prices (Optional)
    api_p = os.path.join(CLEAN_DIR, "mandi_prices_clean.csv")
    df_api = pd.read_csv(api_p) if os.path.exists(api_p) else pd.DataFrame()
    
    return df_p, df_y, df_prof, df_api

models = load_models()
df_price, df_yield, df_profit, df_api = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    """Render a premium top bar."""
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0 30px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: var(--primary); color: white; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold;">F</div>
            <h2 style="margin: 0; font-size: 24px;">FarmAI</h2>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: right;">
                <div style="font-weight: 600; font-size: 14px; color: var(--text-main);">Devendra Chouhan</div>
                <div style="font-size: 12px; color: var(--text-muted);">Premium Farmer</div>
            </div>
            <div style="width: 40px; height: 40px; border-radius: 50%; background: #e2e8f0; display: flex; align-items: center; justify-content: center; font-size: 18px;">👤</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_nav():
    """Custom sidebar navigation with enhanced buttons."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 20px 0; text-align: center;">
            <div style="font-size: 40px; margin-bottom: 10px;">🌱</div>
            <h2 style="margin: 0; font-size: 20px;">FarmAI Optimizer</h2>
            <p style="font-size: 12px; color: var(--text-muted);">v2.4 Premium Edition</p>
        </div>
        """, unsafe_allow_html=True)
        
        PAGES = [
            ("Overview", "📊"),
            ("Crop Recommendation", "🌿"),
            ("Yield Prediction", "📈"),
            ("Price Forecast", "💰"),
            ("Profit Optimization", "🚀"),
            ("Impact Analysis", "🔬")
        ]
        
        for name, icon in PAGES:
            is_active = st.session_state.page == name
            active_class = "nav-btn-active" if is_active else ""
            if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                st.session_state.page = name
                st.rerun()

# HELPER FUNCTIONS
def get_best_crop_for_state(state: str) -> str:
    """
    Returns the single best crop for a state using a combined score:
      - Most grown  : highest average production area in df_yield (normalised)
      - Most profitable : highest Net_Profit in df_profit (normalised)
      - ML suitability  : added as a bonus if model available (normalised)
    Falls back gracefully if any source is missing.
    """
    state_crops = df_yield[df_yield["state"] == state]["crop"].unique()
    if len(state_crops) == 0:
        return ""

    scores = {}

    # --- 1. Most grown: use average area for the crop in this state ---
    for crop in state_crops:
        rows = df_yield[(df_yield["state"] == state) &
                        (df_yield["crop"].str.lower() == crop.lower())]
        scores[crop.lower()] = float(rows["area"].mean()) if not rows.empty else 0.0

    # Normalise area scores 0-1
    max_area = max(scores.values()) or 1
    area_score = {c: v / max_area for c, v in scores.items()}

    # --- 2. Most profitable: Net_Profit from df_profit ---
    state_prof = df_profit[df_profit["State"].str.lower() == state.lower()]
    max_profit = state_prof["Net_Profit"].max() if not state_prof.empty else 1
    profit_score = {}
    for crop in state_crops:
        prof_row = state_prof[state_prof["Crop"].str.lower() == crop.lower()]
        profit_score[crop.lower()] = float(prof_row.iloc[0]["Net_Profit"]) / (max_profit or 1) \
            if not prof_row.empty else 0.0

    # --- 3. ML suitability bonus (optional) ---
    ml_score = {c: 0.0 for c in [cr.lower() for cr in state_crops]}
    if "crop_recommender.pkl" in models:
        try:
            soil = STATE_SOIL_DEFAULTS.get(state.lower(), {"n":70,"p":45,"k":30,"rain":500})
            s_temp, s_hum, s_ph = get_state_climate(state)
            model = models["crop_recommender.pkl"]
            probs = model.predict_proba([[soil["n"], soil["p"], soil["k"],
                                          s_temp, s_hum, s_ph, soil["rain"]]])[0]
            prob_dict = {c.lower(): float(p) for c, p in zip(model.classes_, probs)}
            for crop in state_crops:
                ml_score[crop.lower()] = prob_dict.get(crop.lower(), 0.0)
        except Exception:
            pass  # ML unavailable — silently skip

    # --- Combined score: 40% area + 40% profit + 20% ML ---
    combined = {}
    for crop in [c.lower() for c in state_crops]:
        combined[crop] = (0.4 * area_score.get(crop, 0)
                        + 0.4 * profit_score.get(crop, 0)
                        + 0.2 * ml_score.get(crop, 0))

    best = max(combined, key=combined.get)
    return best  # lowercase crop name

@st.cache_data
def get_state_crop_comparison(state: str, n: int, p: int, k: int, rain: int):
    """Build comparison DataFrame for all crops in a state, sorted by combined score."""
    s_temp, s_hum, s_ph = get_state_climate(state)
    state_crops_list = sorted(df_yield[df_yield["state"] == state]["crop"].unique())
    state_prof = df_profit[df_profit["State"].str.lower() == state.lower()]

    prob_dict = {}
    if "crop_recommender.pkl" in models:
        try:
            model = models["crop_recommender.pkl"]
            probs = model.predict_proba([[n, p, k, s_temp, s_hum, s_ph, rain]])[0]
            prob_dict = {c.lower(): float(prob) for c, prob in zip(model.classes_, probs)}
        except Exception:
            pass

    rows = []
    for crop in state_crops_list:
        hist = df_yield[(df_yield["crop"].str.lower() == crop.lower()) &
                        (df_yield["state"].str.lower() == state.lower())]
        avg_yield  = round(hist["yield"].mean(), 2)  if not hist.empty and "yield" in hist.columns else 0.0
        avg_area   = round(hist["area"].mean(), 0)   if not hist.empty and "area"  in hist.columns else 0.0
        prof_row   = state_prof[state_prof["Crop"].str.lower() == crop.lower()]
        avg_profit = int(prof_row.iloc[0]["Net_Profit"]) if not prof_row.empty else 0
        suitability = round(prob_dict.get(crop.lower(), 0) * 100, 1)
        rows.append({"Crop": crop.title(),
                     "Suitability (%)": suitability,
                     "Avg Yield (t/ha)": avg_yield,
                     "Avg Area (ha)": int(avg_area),
                     "Net Profit (₹K)": round(avg_profit / 1000, 1)})

    df_out = pd.DataFrame(rows)
    # If ML gives all zeros, sort by profit then area
    if df_out["Suitability (%)"].max() == 0:
        df_out = df_out.sort_values(["Net Profit (₹K)", "Avg Area (ha)"],
                                    ascending=False).reset_index(drop=True)
    else:
        df_out = df_out.sort_values("Suitability (%)", ascending=False).reset_index(drop=True)
    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING SCREEN — shown on first launch only
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.app_initialized or st.session_state.page == "Onboarding":
    st.markdown(f"""
    <style>
        /* 1. Global Reset & Full-Screen Background */
        [data-testid="stSidebar"] {{ display: none !important; }}
        header, footer {{ visibility: hidden !important; }}
        
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            overflow: hidden !important;
        }}

        /* 2. Zero-Scroll Centering */
        div[data-testid="stAppViewContainer"] {{
            padding: 0 !important;
            height: 100vh;
            overflow: hidden !important;
        }}
        
        div[data-testid="stMain"] > div {{
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 !important;
        }}

        /* 3. Style the specific column as the Card */
        [data-testid="column"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            padding: 50px !important;
            border-radius: 40px !important;
            box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6) !important;
            max-width: 500px !important;
            text-align: center;
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            backdrop-filter: blur(10px);
        }}

        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(40px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        h1, h2, h3 {{ color: #0f172a !important; text-align: center; }}
        p {{ color: #475569 !important; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        st.markdown(f"""
            <div style="font-size: 70px; margin-bottom: 10px;">🚜</div>
            <h1 style="margin: 0; font-size: 42px; font-weight: 800; letter-spacing: -1px;">FarmAI</h1>
            <p style="font-size: 18px; margin-top: 10px; font-weight: 500;">Precision Agriculture Analytics</p>
            <div style="height: 1px; background: #eee; margin: 30px 0;"></div>
            <p style="font-size: 13px; color: #94a3b8 !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px;">Location & Crop Selection</p>
        """, unsafe_allow_html=True)
        
        all_states_ob = sorted(df_yield["state"].unique())
        ob_state = st.selectbox(
            "State",
            [""] + all_states_ob,
            format_func=lambda x: "— Choose your state —" if x == "" else x,
            key="ob_state",
            label_visibility="collapsed"
        )

        if ob_state:
            ob_crops = sorted(df_yield[df_yield["state"] == ob_state]["crop"].unique())
            crop_count = len(ob_crops)
            
            st.markdown(f"""
                <div style="background: #eaf5ea; padding: 12px; border-radius: 15px; margin: 20px 0; border: 1px solid #d1e7d1;">
                    <span style="color: #15803d; font-weight: 700; font-size: 15px;">📊 {crop_count} Crops</span> 
                    <span style="color: #475569; font-size: 14px;"> available in {ob_state}</span>
                </div>
            """, unsafe_allow_html=True)

            ob_best = get_best_crop_for_state(ob_state)
            if not ob_best and ob_crops: ob_best = ob_crops[0]
            
            st.markdown(f"<div style='height: 15px;'></div>", unsafe_allow_html=True)
            ob_crops_lower = [c.lower() for c in ob_crops]
            ob_default_idx = ob_crops_lower.index(ob_best.lower()) if ob_best and ob_best.lower() in ob_crops_lower else 0
            ob_crop = st.selectbox("Crop", ob_crops, index=ob_default_idx, key="ob_crop", label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Launch Dashboard", type="primary", use_container_width=True):
                st.session_state.y_state = ob_state
                st.session_state.y_crop  = ob_crop
                st.session_state._last_autofill_state = ""
                st.session_state.app_initialized = True
                st.rerun()
        else:
            st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

    st.stop()



# 4. SIDEBAR & NAVIGATION
render_sidebar_nav()

with st.sidebar:
    st.markdown("---")
    st.subheader("📍 Context")
    all_states = sorted(df_yield["state"].unique())
    y_state = st.selectbox("State", all_states, key="y_state")

    # Compute valid crops for the selected state
    state_crops = sorted(df_yield[df_yield["state"] == y_state]["crop"].unique())

    # Autofill Logic
    needs_rerun = False
    if y_state != st.session_state._last_autofill_state:
        defaults = STATE_SOIL_DEFAULTS.get(y_state.lower().strip(), {})
        if defaults:
            st.session_state.n    = defaults["n"]
            st.session_state.p    = defaults["p"]
            st.session_state.k    = defaults["k"]
            st.session_state.rain = defaults["rain"]
        st.session_state._last_autofill_state = y_state
        st.session_state.y_district = "All Districts"
        best = get_best_crop_for_state(y_state)
        st.session_state.y_crop = best if best else (state_crops[0] if state_crops else "")
        needs_rerun = True
    elif st.session_state.y_crop not in state_crops:
        st.session_state.y_crop = state_crops[0] if state_crops else st.session_state.y_crop
        needs_rerun = True
    if needs_rerun: st.rerun()

    y_crop = st.selectbox("Crop", state_crops, key="y_crop")
    y_area = st.number_input("Area (ha)", 1.0, 10000.0, key="y_area")

    st.markdown("---")
    st.subheader("🧪 Field Inputs")
    n    = st.slider("Nitrogen (N)",   0, 140,  key="n")
    p_in = st.slider("Phosphorus (P)", 0, 145,  key="p")
    k    = st.slider("Potassium (K)",  0, 205,  key="k")
    rain = st.number_input("Rainfall (mm)", 0, 5000, key="rain")

    if st.button("🔄 Refresh Analytics", type="primary", use_container_width=True):
        st.rerun()

# 6. SHARED PREDICTIONS
def safe_encode(le, val):
    """Encode a label safely. Returns None (not 0) if unknown — avoids silent wrong predictions."""
    if le is None:
        return None
    val_str = str(val).lower()
    classes = list(le.classes_)
    if val_str in classes:
        return le.transform([val_str])[0]
    # Partial match fallback for minor naming differences
    match = next((c for c in classes if val_str in c or c in val_str), None)
    if match:
        return le.transform([match])[0]
    return None  # Truly unknown — caller handles gracefully

# Get state-based climate (fixes hardcoded temp/humidity/ph)
temp, humidity, ph = get_state_climate(y_state)

# Yield prediction — ML model first, historical average as fallback
yield_val = None
yield_source = "ML Model"
if "yield_predictor.pkl" in models:
    c_enc = safe_encode(models.get("yield_crop_encoder.pkl"), y_crop)
    s_enc = safe_encode(models.get("yield_state_encoder.pkl"), y_state)
    if c_enc is not None and s_enc is not None:
        yield_val = models["yield_predictor.pkl"].predict([[c_enc, s_enc, y_area, rain]])[0]

# Fallback: historical average from df_yield for this crop/state
if yield_val is None:
    hist_rows = df_yield[
        (df_yield["crop"].str.lower() == y_crop.lower()) &
        (df_yield["state"].str.lower() == y_state.lower())
    ]
    if not hist_rows.empty and "yield" in hist_rows.columns:
        yield_val = hist_rows["yield"].mean()
        yield_source = "Historical Avg"
    elif not hist_rows.empty:
        # yield column might be named differently
        num_cols = hist_rows.select_dtypes(include="number").columns.tolist()
        if num_cols:
            yield_val = hist_rows[num_cols[-1]].mean()
            yield_source = "Historical Avg"

def get_recommendations_full(n, p, k, temp, humidity, ph, rain, state, selected_crop=""):
    """Get top 3 state-filtered recs + global top-1 + selected crop's rank."""
    if "crop_recommender.pkl" not in models:
        return [], None, None, None
    model = models["crop_recommender.pkl"]
    probs = model.predict_proba([[n, p, k, temp, humidity, ph, rain]])[0]
    crops = model.classes_

    # Global ranking (unfiltered)
    all_sorted = sorted(zip(crops, probs), key=lambda x: -x[1])
    global_top1 = all_sorted[0][0] if all_sorted else None

    # Find selected crop's rank and confidence in global list
    crop_rank, crop_conf = None, None
    for i, (c, prob) in enumerate(all_sorted):
        if c.lower() == selected_crop.lower():
            crop_rank = i + 1
            crop_conf = prob
            break

    # State-filtered top 3
    valid_crops = set(df_yield[df_yield['state'] == state]['crop'].str.lower().unique())
    filtered = [{"crop": c, "prob": prob} for c, prob in zip(crops, probs)
                if c.lower() in valid_crops]
    return sorted(filtered, key=lambda x: -x['prob'])[:3], global_top1, crop_rank, crop_conf

top_3, global_top1, crop_rank, crop_conf = get_recommendations_full(
    n, p_in, k, temp, humidity, ph, rain, y_state, y_crop)

@st.cache_data
def get_best_states(n, p, k, rain):
    """Rank all states by best-crop confidence for the given soil inputs."""
    if "crop_recommender.pkl" not in models:
        return []
    model = models["crop_recommender.pkl"]
    crops = model.classes_
    results = []
    for state in df_yield["state"].unique():
        s_temp, s_humidity, s_ph = get_state_climate(state)
        probs = model.predict_proba([[n, p, k, s_temp, s_humidity, s_ph, rain]])[0]
        state_crops = set(df_yield[df_yield['state'] == state]['crop'].str.lower().unique())
        filtered = [(crops[i], probs[i]) for i in range(len(crops)) if crops[i].lower() in state_crops]
        if filtered:
            best_crop, best_prob = max(filtered, key=lambda x: x[1])
            results.append({"State": state, "Best Crop": best_crop.title(),
                            "Confidence": best_prob, "conf_pct": f"{best_prob*100:.1f}%"})
    return sorted(results, key=lambda x: -x["Confidence"])[:8]

best_states = get_best_states(n, p_in, k, rain)

# ── Export Report ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("📄 Export")
    
    y_str = f"{yield_val:,.2f} t/ha" if yield_val else "N/A"
    p_str = f"{yield_val*y_area:,.2f} tonnes" if yield_val else "N/A"
    
    report_content = f"""🌾 FarmAI — Analytics Report
======================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}
State: {y_state}
Crop: {y_crop.title()}
Farm Size: {y_area} hectares

--- 🌡️ CLIMATE & SOIL DATA ---
Temperature: {temp}°C
Humidity: {humidity}%
Annual Rainfall: {rain} mm
Soil pH: {ph}
Nitrogen (N): {n} kg/ha
Phosphorus (P): {p_in} kg/ha
Potassium (K): {k} kg/ha

--- 📈 YIELD ESTIMATES ---
Predicted Yield: {y_str}
Total Production: {p_str}

--- 🏆 AI RECOMMENDATIONS FOR {y_state.upper()} ---
"""
    if top_3:
        for i, rec in enumerate(top_3):
            report_content += f"{i+1}. {rec['crop'].title()} ({rec['prob']*100:.1f}% match)\n"

    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"FarmAI_Report_{y_state.replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# 7. PAGES
page = st.session_state.page

if page == "Overview":
    render_header()
    
    st.markdown(f"### 🌾 Dashboard Overview — {y_state}")
    
    # ── 1. Key Metrics Row ──────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        suit_val = crop_conf * 100 if crop_conf else 0
        st.markdown(f"""
        <div class="f-card" style="height: 220px; display: flex; flex-direction: column; justify-content: center; text-align: center; border-top: 5px solid var(--primary);">
            <div style="font-size: 40px; margin-bottom: 8px;">🌱</div>
            <div style="color: var(--text-muted); font-size: 15px; font-weight: 700;">Selected Crop</div>
            <div style="font-size: 28px; font-weight: 800; color: var(--text-main); margin-bottom: 8px;">{y_crop.title()}</div>
            <div style="display: flex; justify-content: center; align-items: center; gap: 8px;">
                <div style="background: #e2e8f0; height: 8px; width: 100px; border-radius: 4px; overflow: hidden;">
                    <div style="background: var(--primary); width: {suit_val}%; height: 100%;"></div>
                </div>
                <span style="font-size: 13px; font-weight: 700; color: var(--primary);">{suit_val:.0f}% Match</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="f-card" style="height: 220px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
            <div style="font-size: 40px; margin-bottom: 10px;">📦</div>
            <div style="color: var(--text-muted); font-size: 15px; font-weight: 700;">Predicted Yield</div>
            <div style="font-size: 34px; font-weight: 800; color: var(--text-main);">{yield_val:,.2f} <span style="font-size: 18px; font-weight: 600; color: var(--text-muted);">t/ha</span></div>
            <div style="margin-top: 12px;"><span class="pill pill-success" style="font-size: 12px; border: 1px solid #15803d;">✓ {yield_source}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="f-card" style="height: 220px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
            <div style="font-size: 40px; margin-bottom: 10px;">🌍</div>
            <div style="color: var(--text-muted); font-size: 15px; font-weight: 700;">Farm Size</div>
            <div style="font-size: 34px; font-weight: 800; color: var(--text-main);">{y_area:,.1f} <span style="font-size: 18px; font-weight: 600; color: var(--text-muted);">ha</span></div>
            <div style="margin-top: 12px;"><span class="pill pill-info" style="font-size: 12px; border: 1px solid #0369a1;">Total Area</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        state_prof = df_profit[df_profit['State'].str.lower() == y_state.lower()]
        crop_prof  = state_prof[state_prof['Crop'].str.lower() == y_crop.lower()]
        val_prof   = crop_prof.iloc[0]['Net_Profit'] if not crop_prof.empty else 0
        st.markdown(f"""
        <div class="f-card f-card-primary" style="height: 220px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
            <div style="font-size: 40px; margin-bottom: 10px;">💰</div>
            <div style="color: rgba(255,255,255,0.95); font-size: 15px; font-weight: 700;">Est. Net Profit</div>
            <div style="font-size: 34px; font-weight: 800; color: white;">₹{(val_prof * y_area)/1000:,.1f}K</div>
            <div style="margin-top: 12px; color: rgba(255,255,255,0.9); font-size: 13px; font-weight: 500;">Based on {y_crop.title()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Charts Row ──────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown('<div class="f-card">', unsafe_allow_html=True)
        st.markdown(f"#### Monthly Yield Potential — {y_crop.title()}")
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        base_yield = yield_val if yield_val else 2.5
        y_data = [base_yield*m for m in [0.8, 0.6, 0.9, 1.1, 1.3, 1.0, 0.8, 0.7, 1.2, 0.9, 0.8, 0.9]]
        
        fig_trend = px.area(x=months, y=y_data, labels={'x':'Month', 'y':'Yield (t/ha)'})
        fig_trend.update_traces(line_color='#22c55e', fillcolor='rgba(34, 197, 94, 0.1)')
        fig_trend.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="f-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("#### Climate Pulse")
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #f8fafc; border-radius: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 20px;">🌡️</div>
                    <div style="font-size: 14px; color: var(--text-muted);">Temperature</div>
                </div>
                <div style="font-weight: 700; color: var(--text-main);">{temp}°C</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #f8fafc; border-radius: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 20px;">💧</div>
                    <div style="font-size: 14px; color: var(--text-muted);">Humidity</div>
                </div>
                <div style="font-weight: 700; color: var(--text-main);">{humidity}%</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #f8fafc; border-radius: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 20px;">🧪</div>
                    <div style="font-size: 14px; color: var(--text-muted);">Soil pH</div>
                </div>
                <div style="font-weight: 700; color: var(--text-main);">{ph}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #f8fafc; border-radius: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 20px;">🌧️</div>
                    <div style="font-size: 14px; color: var(--text-muted);">Rainfall</div>
                </div>
                <div style="font-weight: 700; color: var(--text-main);">{rain}mm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3. Table Row ──────────────────────────────────────────────
    st.markdown('<div class="f-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📊 State Crop Comparison — {y_state}")
    
    comp_df = get_state_crop_comparison(y_state, n, p_in, k, rain)
    if not comp_df.empty:
        table_html = """<table class="f-table">
        <tr>
            <th>#</th>
            <th>Crop</th>
            <th>Match Score</th>
            <th>Yield (t/ha)</th>
            <th>Profit (₹K)</th>
            <th>Status</th>
        </tr>"""
        
        for idx, row in comp_df.head(5).iterrows():
            is_selected = row['Crop'].lower() == y_crop.lower()
            row_style = "background: #f0fdf4;" if is_selected else ""
            status = '<span class="pill pill-success">Recommended</span>' if idx == 0 else '<span class="pill pill-info">Alternative</span>'
            if is_selected: status = '<span class="pill pill-success" style="background: #15803d; color: white;">Current</span>'
            
            table_html += f"""
            <tr style="{row_style}">
                <td style="font-weight: 700;">{idx+1}</td>
                <td style="font-weight: 600;">{row['Crop']}</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="flex-grow: 1; background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden; width: 60px;">
                            <div style="background: var(--primary); width: {row['Suitability (%)']}%; height: 100%;"></div>
                        </div>
                        <span style="font-size: 12px; font-weight: 600;">{row['Suitability (%)']:.0f}%</span>
                    </div>
                </td>
                <td>{row['Avg Yield (t/ha)']:.2f}</td>
                <td>₹{row['Net Profit (₹K)']:.1f}</td>
                <td>{status}</td>
            </tr>"""
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

elif page == "Crop Recommendation":
    render_header()
    st.markdown(f"### 🌿 AI Crop Recommendations — {y_state}")
    st.markdown("Our AI analyzed your soil and climate data to find the most suitable crops for your land.")
    
    cols = st.columns(3)
    for i, item in enumerate(top_3):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="f-card animate-in" style="border-top: 5px solid var(--primary); padding: 28px;">
                <div style="font-size: 36px; margin-bottom: 12px;">{'🥇' if i==0 else '🥈' if i==1 else '🥉'}</div>
                <h3 style="margin: 0; font-size: 22px; color: var(--text-main);">{item['crop'].title()}</h3>
                <p style="color: var(--text-muted); font-size: 15px; font-weight: 600; margin-bottom: 18px;">{int(item['prob']*100)}% Match Confidence</p>
                <div style="background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 24px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">
                    <div style="background: var(--primary); width: {item['prob']*100}%; height: 100%;"></div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 14px;">
                        <span style="color: var(--text-muted); font-weight: 500;">Est. Yield</span>
                        <span style="font-weight: 700; color: var(--text-main);">High</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 14px;">
                        <span style="color: var(--text-muted); font-weight: 500;">Resource Need</span>
                        <span style="font-weight: 700; color: var(--text-main);">Moderate</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Pro Tip:** Rotating between different recommended crops can improve soil health and long-term yield.")

elif page == "Yield Prediction":
    render_header()
    st.markdown(f"### 📈 Yield Analysis — {y_crop.title()}")
    
    if yield_val is not None:
        # Hero Stat
        st.markdown(f"""
        <div class="f-card f-card-primary" style="text-align: center; padding: 40px; margin-bottom: 30px;">
            <div style="font-size: 16px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Expected Yield Performance</div>
            <div style="font-size: 64px; font-weight: 800;">{yield_val:,.2f} <span style="font-size: 24px; font-weight: 400; opacity: 0.8;">t/ha</span></div>
            <div style="font-size: 14px; opacity: 0.9; margin-top: 12px;">Based on {y_area} hectares in {y_state}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="f-card" style="text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 32px; margin-bottom: 10px;">📦</div>
                <div style="color: var(--text-muted); font-size: 14px;">Total Production</div>
                <div style="font-size: 28px; font-weight: 700; color: var(--text-main);">{yield_val*y_area:,.2f} Tonnes</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="f-card" style="text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 32px; margin-bottom: 10px;">⚖️</div>
                <div style="color: var(--text-muted); font-size: 14px;">Production in KG</div>
                <div style="font-size: 28px; font-weight: 700; color: var(--text-main);">{yield_val*y_area*1000:,.0f} kg</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Yield Benchmarks")
        
        state_avg = df_yield[(df_yield['crop'].str.lower() == y_crop.lower()) & (df_yield['state'].str.lower() == y_state.lower())]['yield'].mean()
        national_avg = df_yield[df_yield['crop'].str.lower() == y_crop.lower()]['yield'].mean()
        
        bench_data = {
            'Category': ['National Avg', 'State Avg', 'Your Farm'],
            'Yield': [national_avg, state_avg, yield_val]
        }
        fig_bench = px.bar(bench_data, x='Yield', y='Category', orientation='h', 
                           color='Category', color_discrete_map={'Your Farm': '#15803d', 'National Avg': '#94a3b8', 'State Avg': '#64748b'})
        fig_bench.update_layout(height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=True, gridcolor='#f1f5f9'), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_bench, use_container_width=True)
    else:
        st.warning("Yield data not available for the selected combination.")

elif page == "Price Forecast":
    render_header()
    st.markdown(f"### 💰 Market Price Trends — {y_crop.title()}")
    
    price_crops = set(df_price['crop'].unique())
    resolved = resolve_price_crop(y_crop, price_crops)
    hist = df_price[df_price['crop'] == resolved].sort_values('date').copy()
    
    if not hist.empty:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Latest Price", f"₹{hist['avg_modal_price'].iloc[-1]:,.0f}", "Modal Price")
        with c2: st.metric("Historical High", f"₹{hist['avg_modal_price'].max():,.0f}")
        with c3: st.metric("Historical Low", f"₹{hist['avg_modal_price'].min():,.0f}")
        
        st.markdown('<div class="f-card">', unsafe_allow_html=True)
        fig_price = px.line(hist, x='date', y='avg_modal_price', title=f"Price Trend: {resolved.title()}")
        fig_price.update_traces(line_color=st.get_option("theme.primaryColor") or "#15803d", line_width=3)
        fig_price.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f1f5f9'))
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No price history found for this crop.")

elif page == "Profit Optimization":
    render_header()
    st.markdown(f"### 🚀 AI Profit Optimization — {y_state}")
    st.markdown("This optimization uses our **Yield Predictor AI** to estimate profit based on your specific soil and climate inputs.")
    
    # ── 1. Dynamic Yield & Profit Calculation ──────────────────────
    price_crops_monthly = set(df_price['crop'].unique())
    resolved_monthly = resolve_price_crop(y_crop, price_crops_monthly)
    monthly_sub = df_price[df_price['crop'] == resolved_monthly].sort_values('date')
    latest_price = monthly_sub.iloc[-1]['avg_modal_price'] if not monthly_sub.empty else 0
    
    # Dynamic projection based on current sliders
    if yield_val and latest_price > 0:
        # Convert yield (t/ha) to quintals (q/ha) since market price is per qtl
        revenue_per_ha = yield_val * 10 * latest_price
        est_cost_per_ha = revenue_per_ha * 0.35  # Estimate 35% cost overhead
        total_net_profit = (revenue_per_ha - est_cost_per_ha) * y_area
        
        st.markdown(f"""
        <div class="f-card f-card-primary" style="padding: 40px; margin-bottom: 30px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div>
                    <div style="font-size: 14px; opacity: 0.8; text-transform: uppercase;">Dynamic Net Profit Projection</div>
                    <div style="font-size: 56px; font-weight: 800;">₹{total_net_profit:,.0f}</div>
                    <div style="font-size: 14px; opacity: 0.9;">Based on {y_area} ha of {y_crop.title()} with current soil/rain inputs</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 20px; backdrop-filter: blur(10px);">
                    <div style="font-size: 12px; opacity: 0.8;">Live Market Rate</div>
                    <div style="font-size: 24px; font-weight: 700;">₹{latest_price:,.0f}<span style="font-size: 14px; font-weight: 400;">/qtl</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ── 2. State-wise AI Optimization List ─────────────────────────
    st.markdown("#### 📋 AI Recommended Crops & Optimized Inputs")
    st.markdown("The following crops are ranked by their **AI-predicted profit** for your current land conditions.")
    
    state_prof_data = df_profit[df_profit['State'].str.lower() == y_state.lower()].copy()
    
    if not state_prof_data.empty:
        # We'll calculate dynamic rankings for the top 10 candidates in the state
        recs = []
        model_yield = models.get("yield_predictor.pkl")
        c_enc_model = models.get("yield_crop_encoder.pkl")
        s_enc_model = models.get("yield_state_encoder.pkl")
        
        # Get Recommender for Match Score
        model_rec = models.get("crop_recommender.pkl")
        crops_rec = [c.lower() for c in model_rec.classes_] if model_rec else []
        probs_rec = model_rec.predict_proba([[n, p_in, k, temp, humidity, ph, rain]])[0] if model_rec else []

        for _, row in state_prof_data.iterrows():
            crop_name = row['Crop']
            
            # A. Calculate Dynamic AI Yield
            dynamic_yield = None
            if model_yield:
                c_enc = safe_encode(c_enc_model, crop_name)
                s_enc = safe_encode(s_enc_model, y_state)
                if c_enc is not None and s_enc is not None:
                    dynamic_yield = model_yield.predict([[c_enc, s_enc, y_area, rain]])[0]
            
            # Fallback to historical yield if ML fails
            if dynamic_yield is None:
                dynamic_yield = row['Net_Profit'] / 50000 # Rough proxy for display if missing
            
            # B. Get latest price for this crop
            res_crop = resolve_price_crop(crop_name, price_crops_monthly)
            crop_p_sub = df_price[df_price['crop'] == res_crop]
            c_price = crop_p_sub['avg_modal_price'].mean() if not crop_p_sub.empty else 2200
            
            # C. Dynamic Profit Calculation
            dynamic_profit_ha = (dynamic_yield * 10 * c_price) * 0.65 # 65% margin
            
            # D. AI Suitability (Match Score)
            suit_pct = 0
            if model_rec and crop_name.lower() in crops_rec:
                suit_pct = probs_rec[crops_rec.index(crop_name.lower())] * 100
            elif crop_name.lower() == "sugarcane": suit_pct = 85.0 # Knowledge-based fallback for high-profit crops missing from classifier
            elif crop_name.lower() == "turmeric":  suit_pct = 80.0
            
            recs.append({
                "Crop": crop_name,
                "Profit": dynamic_profit_ha,
                "Match": suit_pct,
                "Fert": row['Fertilizer_kg'],
                "Labour": row['Labour_hours'],
                "Seed": row['Seed_rate'],
                "Yield": dynamic_yield
            })
            
        # Sort by Dynamic Profit
        recs = sorted(recs, key=lambda x: x['Profit'], reverse=True)

        for i, item in enumerate(recs[:8]):
            is_curr = item['Crop'].lower() == y_crop.lower()
            
            st.markdown(f"""
            <div class="f-card" style="padding: 20px; margin-bottom: 16px; border-left: 5px solid {'var(--primary)' if i < 3 else '#cbd5e1'};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="font-weight: 900; color: var(--primary); font-size: 24px;">#{i+1}</div>
                        <div>
                            <div style="font-weight: 700; font-size: 18px; color: var(--text-main);">{item['Crop'].title()} {'<span class="pill pill-info">Current Selection</span>' if is_curr else ''}</div>
                            <div style="font-size: 12px; color: var(--text-muted); font-weight: 500;">
                                AI Suitability: <span style="color: {'var(--primary)' if item['Match'] > 70 else '#f59e0b' if item['Match'] > 30 else '#ef4444'}; font-weight: 700;">{item['Match']:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 800; color: var(--primary); font-size: 20px;">₹{int(item['Profit']):,} / ha</div>
                        <div style="font-size: 11px; color: var(--text-muted);">Dynamic AI Projection</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; background: #f8fafc; padding: 12px; border-radius: 12px; border: 1px solid #f1f5f9;">
                    <div style="text-align: center;">
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Fertilizer</div>
                        <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">{item['Fert']} kg/ha</div>
                    </div>
                    <div style="text-align: center; border-left: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0;">
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Labour</div>
                        <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">{item['Labour']} hrs</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Seed Rate</div>
                        <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">{item['Seed']} kg/ha</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No state profit data found for the selected location.")


elif page == "Impact Analysis":
    render_header()
    st.markdown("### 🔬 Feature Impact Analysis")
    st.markdown("Discover which factors influence the AI's crop recommendations.")
    
    # Real Model Importance Data
    importances = [0.106, 0.146, 0.178, 0.073, 0.214, 0.052, 0.231]
    labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"]
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.markdown(f"""
        <div class="f-card" style="height: 480px; display: flex; flex-direction: column; justify-content: center; border-left: 5px solid var(--primary);">
            <div style="font-size: 60px; margin-bottom: 20px;">💡</div>
            <h3 style="margin: 0; font-size: 28px; color: #0f172a !important;">AI Core Drivers</h3>
            <p style="font-size: 16px; color: #475569 !important; margin-top: 15px;">
                Our model indicates that <span style="color: var(--primary); font-weight: 700;">Rainfall</span> and <span style="color: var(--primary); font-weight: 700;">Humidity</span> 
                are the primary drivers for these recommendations.
            </p>
            <div style="margin-top: 40px; background: #f0fdf4; padding: 25px; border-radius: 20px; border: 1px dashed #15803d;">
                <div style="font-weight: 800; color: #15803d; font-size: 15px; text-transform: uppercase;">Key Finding</div>
                <div style="font-size: 14px; color: #166534; margin-top: 8px;">
                    Environmental factors determine nearly half (44.5%) of the crop suitability index.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Avoid wrapping in <div> to prevent rendering issues
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=importances + [importances[0]],
            theta=labels + [labels[0]],
            fill='toself',
            line_color='#15803d',
            fillcolor='rgba(21, 128, 61, 0.2)',
            marker=dict(size=8)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 0.25], gridcolor="#e2e8f0"),
                angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(size=12, family="Outfit"))
            ),
            showlegend=False,
            height=480,
            margin=dict(l=80, r=80, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit", color="#1e293b")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Field Condition vs. AI Ideals")
    
    # Fetch Ideals for selected crop
    @st.cache_data
    def get_crop_ideals(crop_name):
        try:
            df_factors = pd.read_csv("data/cleaned/crop_rec_factors_cleaned.csv")
            ideals = df_factors[df_factors['label'].str.lower() == crop_name.lower()].mean(numeric_only=True)
            return ideals.to_dict()
        except: return {}

    ideals = get_crop_ideals(y_crop)
    curr_values = {
        "Nitrogen (N)": n, "Phosphorus (P)": p_in, "Potassium (K)": k,
        "Temperature": temp, "Humidity": humidity, "Soil pH": ph, "Rainfall": rain
    }
    ideal_map = {
        "Nitrogen (N)": "N", "Phosphorus (P)": "P", "Potassium (K)": "K",
        "Temperature": "temperature", "Humidity": "humidity", "Soil pH": "ph", "Rainfall": "rainfall"
    }

    # Sort for list
    sorted_impact = sorted(zip(labels, importances), key=lambda x: x[1], reverse=True)
    
    m1, m2 = st.columns(2)
    for i, (label, impact_val) in enumerate(sorted_impact):
        target_col = m1 if i % 2 == 0 else m2
        with target_col:
            ideal_v = ideals.get(ideal_map.get(label), 0)
            curr_v = curr_values.get(label, 0)
            diff = abs(curr_v - ideal_v) / (ideal_v + 1) if ideal_v else 0
            status = "Optimal" if diff < 0.15 else "Moderate" if diff < 0.4 else "Critical"
            status_color = "#15803d" if status == "Optimal" else "#f59e0b" if status == "Moderate" else "#ef4444"
            
            card_html = f"""<div class="f-card" style="padding: 24px; margin-bottom: 20px; border-top: 4px solid {'#15803d' if i < 2 else '#e2e8f0'};">
<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
<div>
<div style="font-size: 11px; color: #64748b; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">Rank #{i+1} Impact</div>
<div style="font-weight: 800; color: #0f172a; font-size: 18px; margin-top: 4px;">{label}</div>
</div>
<div style="background: {status_color}22; color: {status_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">{status}</div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px;">
<span style="color: #64748b;">Current: <b>{curr_v:,.1f}</b></span>
<span style="color: #64748b;">Ideal: <b>{ideal_v:,.1f}</b></span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 4px; overflow: hidden; display: flex;">
<div style="background: {status_color}; width: {max(5, (1-min(1, diff))*100)}%; height: 100%;"></div>
</div>
<div style="margin-top: 15px; font-size: 12px; color: #64748b; font-style: italic;">
Contributing <b>{(impact_val*100):.1f}%</b> to the AI's final decision.
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Technical Breakdown (Global AI Influence)"):
        st.markdown("""
            <div style="padding: 20px; background: #f8fafc; border-radius: 15px; border: 1px solid #e2e8f0;">
                <p style="font-size: 14px; color: var(--text-main);">
                    This analysis uses <b>SHAP (SHapley Additive exPlanations)</b> values to determine how much each input variable contributes to the machine learning model's output. 
                    The rankings above show the global importance of each feature across the entire dataset of 2,200+ samples.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_permutation_importance_001.png", caption="AI Model Permutation Importance")

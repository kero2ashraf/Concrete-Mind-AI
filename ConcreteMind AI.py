"""
ConcreteMind AI Platform — Streamlit + OpenRouter
4 specialist agents:
  QC Agent     → meta-llama/llama-3.3-70b-instruct:free
  Mix Design   → deepseek/deepseek-v4-flash:free
  Product      → google/gemma-4-31b-it:free
  Docs Agent   → nvidia/nemotron-3-super-120b-a12b:free

Speed features:
  • Streaming (word-by-word output)
  • max_tokens capped at 1200
  • Fast fallback chain
"""

import re as _re
import streamlit as st
import requests
from datetime import datetime

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConcreteMind AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "qc_history":   [],
        "mix_history":  [],
        "prod_history": [],
        "docs_history": [],
        "active_tab":   "Dashboard",
        "api_key":      "",
        "qc_prod": "", "qc_batch": "", "qc_cs": 0.0, "qc_cs7": 0.0,
        "qc_slump": 0, "qc_wc": 0.0, "qc_air": 0.0, "qc_temp": 25.0, "qc_notes": "",
        "m_name": "", "m_str": 30, "m_adm": "", "m_spec": "",
        "p_spec": "", "p_desc": "", "p_q": "",
        "d_co": "", "d_notes": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── constants ──────────────────────────────────────────────────────────────────
REGIONS = {
    "UAE / Gulf":     ["ACI 318", "BS 8500", "EN 206"],
    "Egypt":          ["ECP 203", "ASTM C39", "EN 206"],
    "Saudi Arabia":   ["ACI 318", "BS 8500", "SASO"],
    "European Union": ["EN 206", "Eurocode 2", "EN 12390"],
    "United Kingdom": ["BS 8500", "BS EN 206", "EN 12390"],
    "United States":  ["ACI 318", "ACI 211.1", "ASTM C39"],
    "India":          ["IS 456", "IS 10262", "IS 516"],
    "China":          ["GB 50010", "GB/T 50081", "GB 50204"],
    "Australia / NZ": ["AS 1379", "AS 3600", "AS 1012"],
    "Canada":         ["CSA A23.1", "CSA A23.2", "CSA A23.3"],
    "Brazil":         ["NBR 6118", "NBR 12655", "NBR 5739"],
    "Japan":          ["JIS A 5308", "JIS A 1108", "AIJ"],
}

# ── Models ─────────────────────────────────────────────────────────────────────
AGENT_MODELS = {
    "QC Agent":   "meta-llama/llama-3.3-70b-instruct:free",
    "Mix Design": "deepseek/deepseek-v4-flash:free",
    "Product":    "google/gemma-4-31b-it:free",
    "Docs Agent": "meta-llama/llama-3.3-70b-instruct:free",
}
AGENT_LABELS = {
    "QC Agent":   "llama-3.3-70b",
    "Mix Design": "deepseek-v4-flash",
    "Product":    "gemma-4-31b",
    "Docs Agent": "llama-3.3-70b",
}
FALLBACK_CHAIN = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

CONCRETE_CLASSES = ["C15", "C20", "C25", "C30", "C35", "C40", "C45", "C50"]
STRENGTH_MAP = {c: int(c[1:]) for c in CONCRETE_CLASSES}

# ── Cement catalogue ───────────────────────────────────────────────────────────
CEMENT_TYPES = {
    "── Portland Cements (CEM I / ASTM)": [
        "OPC 42.5 N  — CEM I (General purpose)",
        "OPC 42.5 R  — CEM I (Rapid strength)",
        "OPC 52.5 N  — CEM I (High strength)",
        "OPC 52.5 R  — CEM I (High strength rapid)",
        "ASTM Type I  — General purpose",
        "ASTM Type II — Moderate sulfate resistance",
        "ASTM Type III — High early strength",
        "ASTM Type IV — Low heat of hydration",
        "ASTM Type V  — High sulfate resistance",
    ],
    "── Blended / Composite (CEM II–V)": [
        "CEM II/A-S  — Portland slag (6–20% slag)",
        "CEM II/B-S  — Portland slag (21–35% slag)",
        "CEM II/A-V  — Portland fly ash (6–20% FA)",
        "CEM II/B-V  — Portland fly ash (21–35% FA)",
        "CEM II/A-LL — Portland limestone (6–20% LS)",
        "CEM II/B-LL — Portland limestone (21–35% LS)",
        "CEM III/A   — Blast furnace slag (36–65%)",
        "CEM III/B   — Blast furnace slag (66–80%)",
        "CEM IV/A    — Pozzolanic cement (11–35%)",
        "CEM V/A     — Composite cement",
    ],
    "── Special & Regional": [
        "SRC — Sulfate Resistant Cement (BS / ASTM V)",
        "SRPC 42.5   — Sulfate resistant (Gulf/UAE)",
        "HAC / CAC   — High Alumina Cement",
        "White OPC   — White Portland Cement",
        "OWC — Oil Well Cement (Class G / H)",
        "Low Heat Portland Cement (LHPC)",
        "Rapid Hardening Portland Cement (RHPC)",
        "PPC — Portland Pozzolana Cement (IS 1489)",
        "PSC — Portland Slag Cement (IS 455)",
        "Micro-Silica / Silica Fume blended",
        "GGBS blended (Ground Granulated Blast Furnace Slag)",
        "Fly Ash blended (Class F)",
        "Fly Ash blended (Class C)",
        "Metakaolin blended",
        "Geopolymer / Alkali-Activated Cement",
    ],
    "── Gulf / UAE Market Brands": [
        "Emirates Cement — OPC 42.5 N",
        "Emirates Cement — SRC 42.5",
        "Union Cement (UCC) — OPC 52.5",
        "Union Cement (UCC) — SRC",
        "Sharjah Cement — OPC 42.5",
        "Fujairah Cement — OPC 42.5",
        "RAK Cement — OPC / SRC",
        "Al Ain Cement — OPC 42.5",
        "BSCC (Bahrain) — OPC / SRC",
        "Qatari Cement — OPC / SRC",
        "Saudi Cement — OPC 42.5 / 52.5",
        "Lafarge Arabia — CEM I / CEM II",
        "Holcim UAE — CEM I / CEM II / CEM III",
        "Titan Cement — OPC / SRC",
    ],
}
CEMENT_OPTIONS = ["— Select a cement type —"]
for _grp, _items in CEMENT_TYPES.items():
    CEMENT_OPTIONS.append(_grp)
    CEMENT_OPTIONS.extend(_items)
CEMENT_OPTIONS.append("✏️ Custom / Other — type below")

DOC_FIELDS = {
    "QC test report": [
        "Product name", "Batch ID", "Date of test", "Concrete grade",
        "Test lab", "Technician", "CS 7d MPa", "CS 28d MPa",
        "Required strength", "Slump mm", "W/C ratio", "Temperature", "Verdict",
    ],
    "Batch production record": [
        "Batch ID", "Production date", "Shift", "Operator", "Mixer ID",
        "Cement kg", "Water kg", "Fine agg kg", "Coarse agg kg",
        "Admixture kg", "Volume m3", "Target grade",
    ],
    "Technical data sheet": [
        "Product name", "Product code", "Concrete class", "Density kg/m3",
        "Compressive strength", "Slump range", "Max agg size",
        "Applications", "Standards", "Storage",
    ],
    "Delivery ticket": [
        "Ticket no", "Date", "Customer name", "Project name", "Site address",
        "Concrete grade", "Volume m3", "Truck ID", "Driver",
        "Departure time", "Arrival time", "Slump at site",
    ],
    "Non-conformance report": [
        "NCR number", "Date", "Product", "Batch ID", "Non-conformance",
        "Root cause", "Immediate action", "Corrective action",
        "Responsible person", "Target date",
    ],
}

# ── Example data ───────────────────────────────────────────────────────────────
QC_EXAMPLE = {
    "qc_prod": "Ready Mix C30 — Gulf Grade", "qc_batch": "B-2025-042",
    "qc_cs": 27.5, "qc_cs7": 18.2, "qc_slump": 120,
    "qc_wc": 0.48, "qc_air": 1.8, "qc_temp": 38.0,
    "qc_notes": "Hot weather pour, direct sun, no shade on site. Truck delayed 45 min.",
}
MIX_EXAMPLE = {
    "m_name": "MIX-C40-MARINE-UAE", "m_str": 40,
    "m_adm":  "Superplasticizer (BASF MasterGlenium 51), 10% Silica Fume replacement",
    "m_spec": "Coastal tower in Dubai Marina, XS1 exposure, pumped to 35th floor. Max temp 45C during pour.",
}
PROD_EXAMPLE = {
    "p_spec": "Waterproof basement, high early strength for fast cycling",
    "p_desc": "40-floor luxury tower in Dubai Marina. Coastal XS1 exposure. C50 columns, C40 slabs. Pumped concrete required up to 160m. Fast-track schedule needs 3-day striking strength.",
    "p_q": "",
}
DOCS_EXAMPLE = {
    "d_co": "Gulf Ready Mix LLC",
    "fields": {
        "Product name": "Ready Mix C30 Gulf Grade", "Batch ID": "B-2025-042",
        "Date of test": "28/05/2025", "Concrete grade": "C30",
        "Test lab": "Emirates Testing Laboratory, Dubai",
        "Technician": "Eng. Ahmed Al Rashidi",
        "CS 7d MPa": "18.2", "CS 28d MPa": "27.5",
        "Required strength": "30 MPa", "Slump mm": "120",
        "W/C ratio": "0.48", "Temperature": "38C", "Verdict": "FAIL — below required 30 MPa",
    },
    "d_notes": "Hot weather pour, truck delayed 45 minutes. Recommend investigation of w/c ratio.",
}

# ── BEAUTIFUL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

/* ─── ROOT VARIABLES ─────────────────────────────────────────── */
:root {
  --sand:        #C8A96E;
  --sand-light:  #EDD9AA;
  --sand-dim:    #8B7243;
  --ink:         #0F1117;
  --ink2:        #1C1E26;
  --surface:     #F7F5F0;
  --surface2:    #EFECE4;
  --white:       #FFFFFF;
  --border:      rgba(200,169,110,0.22);
  --border2:     rgba(200,169,110,0.45);
  --text:        #1C1E26;
  --muted:       #6B6558;
  --pass-bg:     #F0FAF0;
  --pass-border: #6FCF97;
  --pass-text:   #1A5C2A;
  --fail-bg:     #FDF2F2;
  --fail-border: #EB5757;
  --fail-text:   #7B1D1D;
  --warn-bg:     #FFFBF0;
  --warn-border: #F2C94C;
  --warn-text:   #7B5800;
  --purple:      #5B4FCF;
  --purple-bg:   #EFEDFC;
  --radius:      10px;
}

/* ─── HIDE STREAMLIT CHROME ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ─── KEEP SIDEBAR ALWAYS VISIBLE ───────────────────────────── */
[data-testid="stSidebar"] {
  display: block !important;
  transform: none !important;
  visibility: visible !important;
  min-width: 240px !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  display: block !important;
  transform: none !important;
  width: 240px !important;
  min-width: 240px !important;
}

/* ─── GLOBAL FONT ────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Instrument Sans', sans-serif !important;
  color: var(--text);
}

/* ─── APP BACKGROUND ─────────────────────────────────────────── */
.stApp {
  background: var(--surface) !important;
}
.main .block-container {
  background: transparent;
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px;
}

/* ─── SIDEBAR ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--ink2) !important;
  border-right: 1px solid rgba(200,169,110,0.2) !important;
}
[data-testid="stSidebar"] * {
  color: #D4C9B0 !important;
}
[data-testid="stSidebar"] h3 {
  font-family: 'Syne', sans-serif !important;
  font-size: 20px !important;
  font-weight: 800 !important;
  color: var(--sand-light) !important;
  letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stCaption {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  color: #7A7060 !important;
}
[data-testid="stSidebar"] hr {
  border-color: rgba(200,169,110,0.2) !important;
  margin: 12px 0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
  display: none;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(200,169,110,0.25) !important;
  border-radius: 8px !important;
  color: #D4C9B0 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
  border-color: rgba(200,169,110,0.6) !important;
  box-shadow: 0 0 0 2px rgba(200,169,110,0.12) !important;
}
/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid rgba(200,169,110,0.18) !important;
  color: #A89880 !important;
  border-radius: 8px !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
  text-align: left !important;
  padding: 8px 14px !important;
  margin-bottom: 3px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(200,169,110,0.1) !important;
  border-color: rgba(200,169,110,0.4) !important;
  color: var(--sand-light) !important;
  transform: translateX(2px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #C8A96E, #A07840) !important;
  border-color: transparent !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 12px rgba(200,169,110,0.35) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #D4B87A, #B08850) !important;
  transform: translateX(2px);
  box-shadow: 0 4px 16px rgba(200,169,110,0.45) !important;
}

/* ─── PAGE HEADINGS ──────────────────────────────────────────── */
h1 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  font-size: 2.2rem !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
  margin-bottom: 4px !important;
}
h2 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.3rem !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
}
h3 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  color: var(--ink) !important;
  letter-spacing: 0 !important;
  margin-bottom: 10px !important;
}

/* ─── METRIC CARDS ───────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 20px 18px !important;
  box-shadow: 0 1px 4px rgba(15,17,23,0.06), 0 4px 16px rgba(200,169,110,0.08) !important;
  transition: box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
  box-shadow: 0 2px 8px rgba(15,17,23,0.1), 0 8px 24px rgba(200,169,110,0.12) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 2.2rem !important;
  font-weight: 800 !important;
  color: var(--ink) !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}

/* ─── MAIN BUTTONS ───────────────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #C8A96E 0%, #9A7040 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em !important;
  padding: 10px 24px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 12px rgba(200,169,110,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #D4B87A 0%, #AA8050 100%) !important;
  box-shadow: 0 4px 20px rgba(200,169,110,0.45) !important;
  transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
  background: var(--white) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--surface2) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
}

/* ─── TEXT INPUTS, SELECTS, TEXTAREAS ────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
  background: var(--white) !important;
  border: 1px solid #E2DDD4 !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 13.5px !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div:focus-within {
  border-color: var(--sand) !important;
  box-shadow: 0 0 0 3px rgba(200,169,110,0.15), 0 1px 3px rgba(0,0,0,0.04) !important;
  outline: none !important;
}
.stTextInput label, .stNumberInput label,
.stTextArea label, .stSelectbox label {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
  margin-bottom: 4px !important;
}

/* ─── TABS ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface2) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 7px !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  padding: 8px 20px !important;
  border: none !important;
  transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(200,169,110,0.1) !important;
  color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--white) !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background: transparent !important;
}
.stTabs [data-baseweb="tab-border"] {
  display: none !important;
}

/* ─── EXPANDERS ──────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  padding: 10px 16px !important;
  transition: all 0.15s ease !important;
}
.streamlit-expanderHeader:hover {
  background: var(--surface2) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
}
.streamlit-expanderContent {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
  background: var(--white) !important;
  padding: 16px !important;
}

/* ─── DIVIDERS ───────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 20px 0 !important;
}

/* ─── CAPTIONS ───────────────────────────────────────────────── */
.stCaption {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  color: var(--muted) !important;
}

/* ─── ALERTS ─────────────────────────────────────────────────── */
.stAlert {
  border-radius: 10px !important;
  border: none !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 13.5px !important;
}
[data-baseweb="notification"] {
  border-radius: 10px !important;
}

/* ─── DOWNLOAD BUTTON ────────────────────────────────────────── */
.stDownloadButton > button {
  background: var(--surface2) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
  background: var(--surface) !important;
  border-color: var(--sand) !important;
  color: var(--sand-dim) !important;
}

/* ─── SPINNER ────────────────────────────────────────────────── */
.stSpinner > div {
  border-color: var(--sand) var(--sand) transparent transparent !important;
}

/* ─── SCROLLBAR ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(200,169,110,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200,169,110,0.55); }

/* ─── CUSTOM BADGE CLASSES ───────────────────────────────────── */
.badge-model {
  background: #EEEDFE;
  color: #4C44B0;
  border: 1px solid #C5C0F0;
  padding: 3px 11px;
  border-radius: 20px;
  font-size: 11px;
  font-family: 'DM Mono', monospace;
  font-weight: 500;
  letter-spacing: 0.04em;
}
.badge-pass {
  background: var(--pass-bg);
  color: var(--pass-text);
  border: 1px solid var(--pass-border);
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
}
.badge-fail {
  background: var(--fail-bg);
  color: var(--fail-text);
  border: 1px solid var(--fail-border);
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
}
.badge-warn {
  background: var(--warn-bg);
  color: var(--warn-text);
  border: 1px solid var(--warn-border);
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
}
.sidebar-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #5A5244;
  margin-bottom: 6px;
}
.key-ok {
  background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
  color: #065F46;
  border: 1px solid #6EE7B7;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
}
.key-bad {
  background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
  color: #991B1B;
  border: 1px solid #FCA5A5;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
}
.model-status-ok {
  background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
  color: #065F46;
  border: 1px solid #6EE7B7;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-family: 'DM Mono', monospace;
  font-weight: 600;
  display: inline-block;
}
.verdict-pass {
  background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
  border-left: 4px solid #10B981;
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: #065F46;
  margin: 14px 0;
  box-shadow: 0 2px 12px rgba(16,185,129,0.12);
}
.verdict-fail {
  background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
  border-left: 4px solid #EF4444;
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: #991B1B;
  margin: 14px 0;
  box-shadow: 0 2px 12px rgba(239,68,68,0.12);
}
.verdict-warn {
  background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
  border-left: 4px solid #F59E0B;
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: #92400E;
  margin: 14px 0;
  box-shadow: 0 2px 12px rgba(245,158,11,0.12);
}
.result-card {
  background: var(--white);
  border: 1px solid #E2DDD4;
  border-radius: 12px;
  margin-top: 16px;
  box-shadow: 0 2px 16px rgba(15,17,23,0.07), 0 0 0 1px rgba(200,169,110,0.08);
  overflow: hidden;
}
.result-card-header {
  background: linear-gradient(90deg, #1C1E26 0%, #2C2E3A 100%);
  color: #EDD9AA;
  padding: 12px 20px;
  font-size: 12px;
  font-family: 'DM Mono', monospace;
  letter-spacing: 0.06em;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid rgba(200,169,110,0.2);
}
.result-card-body {
  padding: 20px 24px;
  font-family: 'Instrument Sans', sans-serif;
  font-size: 13.5px;
  line-height: 1.85;
  color: #2A2C38;
  max-height: 560px;
  overflow-y: auto;
}
.result-card-body b,
.result-card-body strong {
  color: #0F1117;
  font-weight: 700;
}
.result-card-body table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0;
  font-size: 12.5px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.result-card-body th {
  background: #1C1E26;
  color: #EDD9AA;
  padding: 10px 12px;
  text-align: left;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.result-card-body td {
  padding: 9px 12px;
  border-bottom: 1px solid #F0EDE6;
  vertical-align: top;
  color: #2A2C38;
}
.result-card-body tr:nth-child(even) td {
  background: #FAF8F4;
}
.result-card-body tr:hover td {
  background: #FDF5E8;
}
.result-card-body h3,
.result-card-body h4 {
  font-family: 'Syne', sans-serif;
  color: #8B6030;
  margin: 18px 0 8px;
  font-size: 12.5px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  border-bottom: 1px solid #F0E8D8;
  padding-bottom: 5px;
}
.result-card-body hr {
  border: none;
  border-top: 1px solid #F0EDE6;
  margin: 14px 0;
}
.stream-box {
  background: #FAF8F4;
  border: 1px solid #E8E2D8;
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'Instrument Sans', sans-serif;
  font-size: 13.5px;
  line-height: 1.85;
  color: #2A2C38;
  min-height: 80px;
  white-space: pre-wrap;
  box-shadow: inset 0 1px 4px rgba(0,0,0,0.04);
}

/* ─── COLUMN CONTAINERS ──────────────────────────────────────── */
[data-testid="column"] > div > div > div {
  background: var(--white);
  border-radius: var(--radius);
  padding: 20px;
  border: 1px solid var(--border);
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  margin-bottom: 2px;
}

/* ─── NUMBER INPUT BUTTONS ───────────────────────────────────── */
.stNumberInput button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
  border-radius: 6px !important;
}
.stNumberInput button:hover {
  background: var(--sand-light) !important;
  color: var(--ink) !important;
}

/* ─── CODE INLINE ────────────────────────────────────────────── */
code {
  background: #F0EBE0 !important;
  color: #7A5C30 !important;
  padding: 2px 7px !important;
  border-radius: 5px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
}

/* ─── INFO / SUCCESS / ERROR BOX ─────────────────────────────── */
[data-baseweb="notification"][kind="info"] {
  background: linear-gradient(135deg, #EFF6FF, #DBEAFE) !important;
  border: 1px solid #BFDBFE !important;
  border-radius: 10px !important;
}
[data-baseweb="notification"][kind="positive"] {
  background: linear-gradient(135deg, #ECFDF5, #D1FAE5) !important;
  border: 1px solid #6EE7B7 !important;
  border-radius: 10px !important;
}
[data-baseweb="notification"][kind="negative"] {
  background: linear-gradient(135deg, #FEF2F2, #FEE2E2) !important;
  border: 1px solid #FCA5A5 !important;
  border-radius: 10px !important;
}

/* ─── MARKDOWN TEXT ──────────────────────────────────────────── */
.stMarkdown p {
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
  color: var(--text) !important;
}
.stMarkdown a {
  color: var(--sand-dim) !important;
  text-decoration: none !important;
  font-weight: 500 !important;
}
.stMarkdown a:hover {
  color: var(--sand) !important;
  text-decoration: underline !important;
}
</style>
""", unsafe_allow_html=True)

# ── Markdown → HTML ────────────────────────────────────────────────────────────
def _fmt(s):
    s = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>', s)
    s = _re.sub(r'`([^`]+)`',     r'<code style="background:#F0EBE0;padding:1px 5px;border-radius:4px;font-size:12px;color:#7A5C30;font-family:DM Mono,monospace;">\1</code>', s)
    return s

def md_to_html(text):
    lines = text.split("\n")
    out = []; in_table = False; header_done = False
    for line in lines:
        if line.strip().startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not in_table:
                in_table = True; header_done = False; out.append("<table>")
            if all(_re.match(r'^[-:]+$', c) for c in cells if c):
                header_done = True; continue
            tag = "th" if not header_done else "td"
            out.append("<tr>" + "".join(f"<{tag}>{_fmt(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</table>"); in_table = False; header_done = False
            s = line.strip()
            if   s.startswith("#### "): out.append(f"<h4>{_fmt(s[5:])}</h4>")
            elif s.startswith("### "):  out.append(f"<h3>{_fmt(s[4:])}</h3>")
            elif s.startswith("## "):   out.append(f"<h3>{_fmt(s[3:])}</h3>")
            elif s in ("---","***","___"): out.append("<hr>")
            elif s == "": out.append("<br>")
            else: out.append(f"<p>{_fmt(s)}</p>")
    if in_table: out.append("</table>")
    return "\n".join(out)

def render_result(result, agent_label, model_used):
    body = md_to_html(result)
    return f"""
<div class='result-card'>
  <div class='result-card-header'>🤖 {agent_label} &nbsp;·&nbsp; <span style='opacity:.7'>{model_used}</span></div>
  <div class='result-card-body'>{body}</div>
</div>"""

# ── STREAMING API call ─────────────────────────────────────────────────────────
def stream_openrouter(agent_key: str, system: str, user: str):
    api_key = st.session_state.api_key
    if not api_key:
        yield "__NO_KEY__"
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://concretemind.app",
        "X-Title":       "ConcreteMind AI",
    }

    primary   = AGENT_MODELS[agent_key]
    all_models = [primary] + [m for m in FALLBACK_CHAIN if m != primary]

    def try_stream(model_id):
        payload = {
            "model":       model_id,
            "temperature": 0.2,
            "max_tokens":  1200,
            "stream":      True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload,
            stream=True, timeout=60,
        )
        if resp.status_code == 429:
            return None, "rate_limited"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return resp, None

    for model_id in all_models:
        resp, err = try_stream(model_id)
        if err:
            if err == "rate_limited":
                st.toast(f"⏳ {model_id.split('/')[-1]} rate-limited — trying next model…", icon="⏳")
                continue
            continue

        full_text = ""
        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    import json
                    chunk = json.loads(line)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning") or ""
                    if content:
                        full_text += content
                        yield content
                except Exception:
                    continue
        except Exception as e:
            continue

        if full_text.strip():
            yield f"__DONE__{model_id}"
            return

    yield "__FAILED__"


def call_openrouter_fast(agent_key: str, system: str, user: str):
    api_key = st.session_state.api_key
    if not api_key:
        return None, "no_key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://concretemind.app",
        "X-Title":       "ConcreteMind AI",
    }

    primary    = AGENT_MODELS[agent_key]
    all_models = [primary] + [m for m in FALLBACK_CHAIN if m != primary]

    import json, time

    def try_one_stream(model_id):
        payload = {
            "model": model_id, "temperature": 0.2, "max_tokens": 1200, "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=60,
        )
        return r

    for model_id in all_models:
        try:
            r = try_one_stream(model_id)
        except Exception:
            continue

        if r.status_code == 429:
            st.toast(f"⏳ {model_id.split('/')[-1]} rate-limited — rotating…", icon="⏳")
            time.sleep(5)
            continue
        if r.status_code != 200:
            continue

        full_text = ""
        placeholder = st.empty()

        try:
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta   = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning") or ""
                    if content:
                        full_text += content
                        placeholder.markdown(
                            f"<div class='stream-box'>{full_text}▌</div>",
                            unsafe_allow_html=True,
                        )
                except Exception:
                    continue
        except Exception:
            continue

        if full_text.strip():
            placeholder.empty()
            return full_text.strip(), model_id

    return None, "all_models_failed"


def ts():
    return datetime.now().strftime("%d/%m/%Y %H:%M")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  —  uses on_click callback so the sidebar never collapses
# ══════════════════════════════════════════════════════════════════════════════
def _set_tab(t):
    st.session_state.active_tab = t

with st.sidebar:
    st.markdown("### 🏭 ConcreteMind")
    st.caption("AI platform · concrete & cement · streaming")
    st.divider()

    st.markdown('<div class="sidebar-label">Agents</div>', unsafe_allow_html=True)
    for icon, tab in zip(["📊","🔬","⚗️","📦","📄"],
                         ["Dashboard","QC Agent","Mix Design","Product","Docs Agent"]):
        lbl = AGENT_LABELS.get(tab, "")
        btn_label = f"{icon} {tab}" + (f"  `{lbl}`" if lbl else "")
        st.button(
            btn_label,
            key=f"nav_{tab}",
            use_container_width=True,
            type="primary" if st.session_state.active_tab == tab else "secondary",
            on_click=_set_tab,
            args=(tab,),
        )

    st.divider()
    st.markdown('<div class="sidebar-label">Region</div>', unsafe_allow_html=True)
    region = st.selectbox("Region", list(REGIONS.keys()), label_visibility="collapsed")
    stds   = REGIONS[region]
    st.markdown("  ".join([f"`{s}`" for s in stds]))

    st.divider()
    st.markdown('<div class="sidebar-label">OpenRouter API Key</div>', unsafe_allow_html=True)

    def _save_key():
        st.session_state.api_key = st.session_state._key_input.strip()

    st.text_input(
        "API Key", type="password", placeholder="sk-or-v1-...",
        label_visibility="collapsed", key="_key_input",
        value=st.session_state.api_key, on_change=_save_key,
        help="Free key at openrouter.ai",
    )
    if st.session_state.api_key:
        st.markdown('<div class="key-ok">🔑 Key saved — streaming ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="key-bad">⚠️ No key — add yours above</div>', unsafe_allow_html=True)
        st.caption("[Get free key → openrouter.ai](https://openrouter.ai)")

    st.divider()
    st.markdown('<div class="sidebar-label">Active Models (streaming)</div>', unsafe_allow_html=True)
    for icon, name, note in [
        ("🔬 QC",      "llama-3.3-70b",    "fast ⚡"),
        ("⚗️ Mix",    "deepseek-v4-flash", "fast ⚡"),
        ("📦 Product", "gemma-4-31b",       "fast ⚡"),
        ("📄 Docs",    "llama-3.3-70b",    "fast ⚡"),
    ]:
        st.caption(f"{icon} `{name}` — {note}")


def need_key():
    if not st.session_state.api_key:
        st.error("Add your OpenRouter API key in the sidebar first.")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "Dashboard":
    c1, c2 = st.columns([6, 1])
    with c1:
        st.title("📊 Dashboard")
        st.caption(f"**{region}** · {' · '.join(stds)} · ⚡ Streaming enabled")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear all", use_container_width=True):
            for k in ("qc_history","mix_history","prod_history","docs_history"):
                st.session_state[k] = []
            st.rerun()

    qc_h  = st.session_state.qc_history
    tot   = len(qc_h)
    passed = sum(1 for t in qc_h if t.get("status") == "PASS")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("QC Tests",  tot)
    m2.metric("✅ Passed", passed)
    m3.metric("❌ Failed",  tot - passed)
    m4.metric("Pass rate", f"{round(passed/tot*100)}%" if tot else "—")

    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.subheader("🤖 Agents & Models")
        icons_map = {"QC Agent":"🔬","Mix Design":"⚗️","Product":"📦","Docs Agent":"📄"}
        for agent, lbl in AGENT_LABELS.items():
            st.markdown(
                f"{icons_map[agent]} **{agent}** — `{lbl}` "
                f"<span class='model-status-ok'>streaming ⚡</span>",
                unsafe_allow_html=True)
    with cr:
        st.subheader("🌍 Standards by Region")
        for rn, rs in list(REGIONS.items())[:6]:
            st.markdown(f"**{rn}**: {' · '.join(rs)}")

    st.divider()
    st.subheader("🕐 Recent QC Tests")
    if not qc_h:
        st.info("No tests yet — start in the QC Agent tab.")
    else:
        for t in reversed(qc_h[-5:]):
            bc = "badge-pass" if t["status"]=="PASS" else ("badge-warn" if t["status"]=="WARNING" else "badge-fail")
            st.markdown(
                f"**{t['product']}** · Batch `{t['batch']}` · {t['cs']} / {t['req']} MPa · "
                f"`{t['std']}` · <span class='{bc}'>{t['status']}</span> · "
                f"<span class='badge-model'>{t['model']}</span>",
                unsafe_allow_html=True)
            st.caption(t["date"]); st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# QC AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "QC Agent":
    st.title("🔬 QC Agent")
    st.markdown(f"`{region}` · <span class='badge-model'>llama-3.3-70b · streaming ⚡</span>",
                unsafe_allow_html=True)

    tab_new, tab_hist = st.tabs(["🧪 New Test", "📋 History"])

    with tab_new:
        with st.expander("💡 Load example — C30 hot-weather pour UAE", expanded=False):
            if st.button("⚡ Load example", key="qc_load_ex"):
                for k, v in QC_EXAMPLE.items():
                    st.session_state[k] = v
                st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Sample Information")
            qc_prod  = st.text_input("Product name", placeholder="e.g. Ready mix C30", key="qc_prod")
            qc_batch = st.text_input("Batch / Mix ID", placeholder="e.g. B-2025-001", key="qc_batch")
            qc_cls   = st.selectbox("Concrete class", CONCRETE_CLASSES, index=3, key="qc_cls")
            qc_std   = st.selectbox("Standard", stds, key="qc_std")
        with col_b:
            st.subheader("Test Results")
            qc_cs    = st.number_input("CS @ 28d (MPa)*", min_value=0.0, step=0.5, key="qc_cs")
            qc_cs7   = st.number_input("CS @ 7d (MPa) — optional", min_value=0.0, step=0.5, key="qc_cs7")
            qc_slump = st.number_input("Slump (mm)", min_value=0, step=5, key="qc_slump")
            qc_wc    = st.number_input("W/C ratio", min_value=0.0, max_value=1.0, step=0.01, key="qc_wc")
            qc_air   = st.number_input("Air content (%)", min_value=0.0, step=0.1, key="qc_air")
            qc_temp  = st.number_input("Temperature (°C)", step=1.0, key="qc_temp")

        qc_notes = st.text_area("Additional notes", placeholder="Site conditions, curing notes...", key="qc_notes")

        if st.button("⚡ Analyze (streaming)", type="primary", key="qc_run"):
            if need_key(): pass
            elif not qc_prod or not qc_cs:
                st.error("Enter product name and 28d compressive strength.")
            else:
                req = STRENGTH_MAP[qc_cls]
                sys_p = (
                    f"You are a senior QC engineer for concrete and cement. "
                    f"Evaluate strictly against {qc_std} and codes: {', '.join(stds)}. "
                    f"Be concise. Format: VERDICT → ANALYSIS (per parameter) → CORRECTIVE ACTIONS → RECOMMENDATIONS."
                )
                usr_p = (
                    f"Region: {region} | Product: {qc_prod} | Batch: {qc_batch or 'N/A'} | "
                    f"Class: {qc_cls} | Required: {req} MPa | Standard: {qc_std}\n"
                    f"CS(28d)={qc_cs} MPa | CS(7d)={qc_cs7 or 'N/A'} | "
                    f"Slump={qc_slump}mm | W/C={qc_wc} | Air={qc_air}% | Temp={qc_temp}°C\n"
                    f"Notes: {qc_notes or 'None'}"
                )
                with st.spinner("Connecting…"):
                    result, used_model = call_openrouter_fast("QC Agent", sys_p, usr_p)

                if result:
                    status = "PASS" if qc_cs >= req else "FAIL"
                    if "WARNING" in result.upper(): status = "WARNING"
                    short = used_model.split("/")[-1].replace(":free","")
                    if status == "PASS":
                        st.markdown(f"<div class='verdict-pass'>✅ PASS — {qc_cs} MPa ≥ {req} MPa ({qc_std})</div>", unsafe_allow_html=True)
                    elif status == "WARNING":
                        st.markdown(f"<div class='verdict-warn'>⚠️ WARNING — review analysis below</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='verdict-fail'>❌ FAIL — {qc_cs} MPa < {req} MPa ({qc_std})</div>", unsafe_allow_html=True)
                    st.markdown(render_result(result, "QC Agent Analysis", short), unsafe_allow_html=True)
                    st.session_state.qc_history.append({
                        "product": qc_prod, "batch": qc_batch or "N/A",
                        "cs": qc_cs, "req": req, "slump": qc_slump, "wc": qc_wc,
                        "status": status, "std": qc_std, "analysis": result,
                        "date": ts(), "model": short,
                    })
                else:
                    st.error(f"All models failed or rate-limited. Last error: {used_model}\n\nWait 30 sec and retry.")

    with tab_hist:
        _, col_h2 = st.columns([5, 1])
        with col_h2:
            if st.button("🗑 Clear", key="qc_clrhist"):
                st.session_state.qc_history = []; st.rerun()
        if not st.session_state.qc_history:
            st.info("No test history yet.")
        for i, item in enumerate(reversed(st.session_state.qc_history)):
            bc = "badge-pass" if item["status"]=="PASS" else ("badge-warn" if item["status"]=="WARNING" else "badge-fail")
            with st.expander(f"**{item['product']}** — {item['date']} — {item['status']}"):
                st.markdown(
                    f"Batch `{item['batch']}` · {item['cs']} / {item['req']} MPa · "
                    f"`{item['std']}` · <span class='{bc}'>{item['status']}</span> · "
                    f"<span class='badge-model'>{item['model']}</span>", unsafe_allow_html=True)
                st.markdown(render_result(item["analysis"], "QC Agent", item["model"]), unsafe_allow_html=True)
                real = len(st.session_state.qc_history)-1-i
                ca, cb = st.columns([1,5])
                with ca:
                    if st.button("🗑 Del", key=f"qc_del_{i}"):
                        st.session_state.qc_history.pop(real); st.rerun()
                with cb:
                    st.download_button("⬇ .txt", data=item["analysis"],
                        file_name=f"qc_{item['product'].replace(' ','_')}.txt", key=f"qc_dl_{i}")


# ══════════════════════════════════════════════════════════════════════════════
# MIX DESIGN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "Mix Design":
    st.title("⚗️ Mix Design Agent")
    st.markdown(f"`{region}` · <span class='badge-model'>deepseek-v4-flash · streaming ⚡</span>",
                unsafe_allow_html=True)

    tab_new, tab_hist = st.tabs(["🧱 Design New Mix", "📋 History"])

    with tab_new:
        with st.expander("💡 Load example — C40 Marine Mix UAE", expanded=False):
            if st.button("⚡ Load example", key="mix_load_ex"):
                for k, v in MIX_EXAMPLE.items():
                    st.session_state[k] = v
                st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Requirements")
            m_name = st.text_input("Design name / code", placeholder="e.g. MIX-C40-MARINE-UAE", key="m_name")
            m_str  = st.number_input("Target strength (MPa)", step=1, key="m_str")
            m_std  = st.selectbox("Design standard",
                ["ACI 211.1","EN 206","BS 8500","IS 10262","ECP 203","CSA A23.1"], key="m_std")
            m_exp  = st.selectbox("Exposure class", [
                "X0 — no exposure","XC1 — carbonation dry","XC2 — carbonation wet",
                "XS1 — marine airborne","XS2 — marine submerged",
                "XF — freeze-thaw","XA — chemical attack","Hot weather >45°C (Gulf)",
            ], key="m_exp")
            m_app  = st.selectbox("Application", [
                "Ready mix concrete","Precast elements","Foundations",
                "Columns & beams","Slabs","Paving / roads","Marine / coastal","Mass concrete",
            ], key="m_app")

        with col_b:
            st.subheader("Materials")
            m_cem_sel = st.selectbox("Cement type", CEMENT_OPTIONS, key="m_cem_sel")
            _is_hdr = m_cem_sel.startswith("──") or m_cem_sel == "— Select a cement type —"
            _is_cus = m_cem_sel.startswith("✏️")
            if _is_hdr or _is_cus:
                m_cem_custom = st.text_input("Custom cement", placeholder="e.g. Emirates Cement OPC 52.5 R", key="m_cem_custom")
                m_cem = m_cem_custom.strip() or "Not specified"
            else:
                m_cem = m_cem_sel
                _hints = {"SRC":"💡 Sulfate resistant — good for UAE marine.", "HAC":"⚠️ Specialist use only.",
                          "CEM III":"💡 Low heat — good for mass concrete.", "GGBS":"💡 30–70% replacement; reduces heat.",
                          "Silica Fume":"💡 5–15% replacement; improves density."}
                for kw, hint in _hints.items():
                    if kw.lower() in m_cem.lower(): st.caption(hint); break
            m_agg  = st.selectbox("Max aggregate size (mm)", ["10","14","20","25","40"], index=2, key="m_agg")
            m_work = st.selectbox("Workability", [
                "25–50 mm stiff","50–100 mm medium",
                "100–150 mm high","150–200 mm pumped","Flow >200 mm SCC",
            ], index=2, key="m_work")
            m_adm  = st.text_input("Admixtures", placeholder="e.g. Superplasticizer, Silica Fume", key="m_adm")
            m_spec = st.text_area("Special requirements", placeholder="e.g. Hot weather, low heat...", key="m_spec")

        if st.button("⚡ Generate mix design (streaming)", type="primary", key="mix_run"):
            if need_key(): pass
            elif not m_name:
                st.error("Enter a design name.")
            else:
                sys_p = (
                    f"You are an expert concrete mix design engineer. Use {m_std} for {region}. "
                    f"Codes: {', '.join(stds)}. Be concise and numerical. "
                    f"Output: MIX DESIGN TABLE per m³ (cement kg, water kg, fine agg kg, coarse agg kg, "
                    f"admixtures, w/c, air%) then NOTES & RECOMMENDATIONS."
                )
                usr_p = (
                    f"Design: {m_name} | Target: {m_str} MPa | Std: {m_std} | Region: {region}\n"
                    f"Exposure: {m_exp} | App: {m_app}\n"
                    f"Cement: {m_cem} | MaxAgg: {m_agg}mm | Workability: {m_work}\n"
                    f"Admixtures: {m_adm or 'None'} | Special: {m_spec or 'None'}"
                )
                with st.spinner("Connecting…"):
                    result, used_model = call_openrouter_fast("Mix Design", sys_p, usr_p)
                if result:
                    short = used_model.split("/")[-1].replace(":free","")
                    st.markdown(f"<div class='verdict-pass'>✅ Mix design generated — <code>{short}</code></div>",
                                unsafe_allow_html=True)
                    st.markdown(render_result(result, "Mix Design", short), unsafe_allow_html=True)
                    st.download_button("⬇ Download .txt", data=result,
                        file_name=f"mix_{m_name.replace(' ','_')}.txt")
                    st.session_state.mix_history.append({
                        "name": m_name, "str": m_str, "std": m_std,
                        "notes": result, "date": ts(), "model": short,
                    })
                else:
                    st.error(f"All models failed. Wait 30 sec and retry.")

    with tab_hist:
        _, col_h2 = st.columns([5, 1])
        with col_h2:
            if st.button("🗑 Clear", key="mix_clrhist"):
                st.session_state.mix_history = []; st.rerun()
        if not st.session_state.mix_history:
            st.info("No saved designs yet.")
        for i, item in enumerate(reversed(st.session_state.mix_history)):
            with st.expander(f"**{item['name']}** — {item['date']} — {item['str']} MPa"):
                st.markdown(f"`{item['std']}` · <span class='badge-model'>{item['model']}</span>", unsafe_allow_html=True)
                st.markdown(render_result(item["notes"], "Mix Design", item["model"]), unsafe_allow_html=True)
                real = len(st.session_state.mix_history)-1-i
                ca, cb = st.columns([1,5])
                with ca:
                    if st.button("🗑 Del", key=f"mix_del_{i}"):
                        st.session_state.mix_history.pop(real); st.rerun()
                with cb:
                    st.download_button("⬇ .txt", data=item["notes"],
                        file_name=f"mix_{item['name'].replace(' ','_')}.txt", key=f"mix_dl_{i}")


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "Product":
    st.title("📦 Product Agent")
    st.markdown(f"`{region}` · <span class='badge-model'>gemma-4-31b · streaming ⚡</span>",
                unsafe_allow_html=True)

    tab_new, tab_hist = st.tabs(["🏗 Recommend", "📋 History"])

    with tab_new:
        with st.expander("💡 Load example — 40-floor Dubai Marina tower", expanded=False):
            if st.button("⚡ Load example", key="prod_load_ex"):
                for k, v in PROD_EXAMPLE.items():
                    st.session_state[k] = v
                st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Project Details")
            p_type = st.selectbox("Project type", [
                "Residential building","Commercial tower","Industrial facility",
                "Bridge / infrastructure","Road / pavement","Marine / coastal",
                "Underground / basement","Precast factory","Water treatment plant",
            ], key="p_type")
            p_env  = st.selectbox("Environment / exposure", [
                "Normal interior dry","Exterior humid","Coastal / marine",
                "Underground / soil contact","Chemical / industrial",
                "High temperature Gulf","Freeze-thaw cycles",
            ], key="p_env")
            p_load = st.selectbox("Load type", [
                "Normal structural","Heavy industrial","Dynamic / seismic",
                "Pre-stressed","Light / non-structural",
            ], key="p_load")
            p_vol  = st.selectbox("Estimated volume", [
                "Small <50 m³","Medium 50–500 m³","Large 500–5000 m³","Mega >5000 m³",
            ], key="p_vol")
            p_spec = st.text_input("Special requirements",
                placeholder="e.g. Waterproof, high early strength...", key="p_spec")
        with col_b:
            st.subheader("Describe Your Project")
            p_desc = st.text_area("Project description", height=130,
                placeholder="e.g. 40-floor tower in Dubai Marina, XS1 exposure, C50 columns...", key="p_desc")
            p_q    = st.text_area("Or ask directly", height=90,
                placeholder="e.g. What grade for a deep basement in Abu Dhabi marine soil?", key="p_q")

        if st.button("⚡ Recommend product (streaming)", type="primary", key="prod_run"):
            if need_key(): pass
            elif not p_desc and not p_q:
                st.error("Describe your project or ask a direct question.")
            else:
                sys_p = (
                    f"You are a senior technical specialist for a concrete manufacturer in {region}. "
                    f"Be concise. Provide: 1) Recommended product & grade, 2) Technical justification, "
                    f"3) Key performance properties, 4) Durability notes, 5) Alternatives, 6) Application tips."
                )
                usr_p = p_q if p_q else (
                    f"Region: {region} | Project: {p_type} | Env: {p_env} | Load: {p_load} | Vol: {p_vol} | "
                    f"Special: {p_spec or 'None'}\nDescription: {p_desc}"
                )
                with st.spinner("Connecting…"):
                    result, used_model = call_openrouter_fast("Product", sys_p, usr_p)
                if result:
                    short = used_model.split("/")[-1].replace(":free","")
                    st.markdown(f"<div class='verdict-pass'>✅ Recommendation ready — <code>{short}</code></div>",
                                unsafe_allow_html=True)
                    st.markdown(render_result(result, "Product Recommendation", short), unsafe_allow_html=True)
                    st.download_button("⬇ Download .txt", data=result, file_name="product_recommendation.txt")
                    st.session_state.prod_history.append({
                        "title": (p_q or p_desc)[:60], "rec": result,
                        "date": ts(), "model": short,
                    })
                else:
                    st.error("All models failed. Wait 30 sec and retry.")

    with tab_hist:
        _, col_h2 = st.columns([5, 1])
        with col_h2:
            if st.button("🗑 Clear", key="prod_clrhist"):
                st.session_state.prod_history = []; st.rerun()
        if not st.session_state.prod_history:
            st.info("No recommendation history yet.")
        for i, item in enumerate(reversed(st.session_state.prod_history)):
            with st.expander(f"**{item['title']}** — {item['date']}"):
                st.markdown(f"<span class='badge-model'>{item['model']}</span>", unsafe_allow_html=True)
                st.markdown(render_result(item["rec"], "Product Recommendation", item["model"]), unsafe_allow_html=True)
                real = len(st.session_state.prod_history)-1-i
                ca, cb = st.columns([1,5])
                with ca:
                    if st.button("🗑 Del", key=f"prod_del_{i}"):
                        st.session_state.prod_history.pop(real); st.rerun()
                with cb:
                    st.download_button("⬇ .txt", data=item["rec"],
                        file_name="product_rec.txt", key=f"prod_dl_{i}")


# ══════════════════════════════════════════════════════════════════════════════
# DOCS AGENT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "Docs Agent":
    st.title("📄 Docs Agent")
    st.markdown(f"`{region}` · <span class='badge-model'>llama-3.3-70b · streaming ⚡</span>",
                unsafe_allow_html=True)

    tab_new, tab_hist = st.tabs(["📝 Generate", "📋 History"])

    with tab_new:
        with st.expander("💡 Load example — QC Test Report (failed batch)", expanded=False):
            if st.button("⚡ Load example", key="docs_load_ex"):
                st.session_state["d_co"]           = DOCS_EXAMPLE["d_co"]
                st.session_state["d_notes"]        = DOCS_EXAMPLE["d_notes"]
                st.session_state["_docs_ex_fields"] = DOCS_EXAMPLE["fields"]
                st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Document Settings")
            d_type = st.selectbox("Document type", list(DOC_FIELDS.keys()), key="d_type")
            d_co   = st.text_input("Company name", placeholder="Your company name", key="d_co")
            st.subheader("Field Values")
            fields    = DOC_FIELDS.get(d_type, [])
            field_vals = {}
            ex_fields  = st.session_state.pop("_docs_ex_fields", None)
            for f in fields:
                fkey = f"df_{f.replace(' ','_')}"
                if ex_fields is not None and d_type == "QC test report":
                    st.session_state[fkey] = ex_fields.get(f, "")
                field_vals[f] = st.text_input(f, key=fkey)
            d_notes = st.text_area("Additional notes", placeholder="Extra context...", key="d_notes")

        with col_b:
            st.subheader("Generated Document")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Generate document (streaming)", type="primary", key="docs_run"):
                if need_key(): pass
                else:
                    sys_p = (
                        f"You are a technical documentation specialist for concrete manufacturing in {region}. "
                        f"Standards: {', '.join(stds)}. Be concise and professional. "
                        f"Generate a standards-compliant {d_type}. "
                        f"Structure: Header → Purpose → Data Table → Results → Remarks → Signature."
                    )
                    today  = datetime.now().strftime("%d/%m/%Y")
                    filled = "\n".join(f"{k}: {v}" for k, v in field_vals.items() if v)
                    usr_p  = (
                        f"Generate a {d_type} for: {d_co or '[Company]'} — Date: {today} — Region: {region}\n"
                        f"Fields:\n{filled or '(none provided)'}\nNotes: {d_notes or 'None'}"
                    )
                    with st.spinner("Connecting…"):
                        result, used_model = call_openrouter_fast("Docs Agent", sys_p, usr_p)
                    if result:
                        short = used_model.split("/")[-1].replace(":free","")
                        st.markdown(f"<div class='verdict-pass'>✅ Document generated — <code>{short}</code></div>",
                                    unsafe_allow_html=True)
                        st.markdown(render_result(result, f"Document: {d_type}", short), unsafe_allow_html=True)
                        st.download_button("⬇ Download .txt", data=result,
                            file_name=f"{d_type.replace(' ','_')}_{today.replace('/','-')}.txt")
                        st.session_state.docs_history.append({
                            "type": d_type,
                            "title": f"{d_type} — {d_co or '[Company]'} — {today}",
                            "content": result, "date": ts(), "model": short,
                        })
                    else:
                        st.error("All models failed. Wait 30 sec and retry.")

    with tab_hist:
        _, col_h2 = st.columns([5, 1])
        with col_h2:
            if st.button("🗑 Clear", key="docs_clrhist"):
                st.session_state.docs_history = []; st.rerun()
        if not st.session_state.docs_history:
            st.info("No saved documents yet.")
        for i, item in enumerate(reversed(st.session_state.docs_history)):
            with st.expander(f"**{item['title']}** — {item['date']}"):
                st.markdown(f"<span class='badge-model'>{item['model']}</span>", unsafe_allow_html=True)
                st.markdown(render_result(item["content"], f"Document: {item['type']}", item["model"]), unsafe_allow_html=True)
                real = len(st.session_state.docs_history)-1-i
                ca, cb = st.columns([1,5])
                with ca:
                    if st.button("🗑 Del", key=f"docs_del_{i}"):
                        st.session_state.docs_history.pop(real); st.rerun()
                with cb:
                    st.download_button("⬇ .txt", data=item["content"],
                        file_name=f"doc_{i}.txt", key=f"docs_dl_{i}")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.cm-footer {
    margin-top: 60px;
    padding: 28px 0 18px 0;
    border-top: 1px solid rgba(200,169,110,0.25);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    text-align: center;
}
.cm-footer-brand {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #1C1E26;
    letter-spacing: 0.04em;
}
.cm-footer-name {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #8B6030;
    letter-spacing: 0.06em;
}
.cm-footer-divider {
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #C8A96E, #8B6030);
    border-radius: 2px;
    margin: 2px auto;
}
.cm-footer-copy {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #9A8F7E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
</style>

<div class="cm-footer">
    <div class="cm-footer-brand">🏭 ConcreteMind AI Platform</div>
    <div class="cm-footer-divider"></div>
    <div class="cm-footer-name">Made by Eng. Kirollos Ashraf Fakhry</div>
    <div class="cm-footer-copy">© 2025 · All rights reserved</div>
</div>
""", unsafe_allow_html=True)

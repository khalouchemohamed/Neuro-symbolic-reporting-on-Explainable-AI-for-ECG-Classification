"""
ECG XAI Pipeline – Streamlit Dashboard
=======================================
A showcase front-end for the Clinically Guided ECG XAI Pipeline.
Provides pipeline execution, output visualization, and report browsing.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
XAI_DIR = OUTPUT_DIR / "xai"
MEDICAL_DIR = OUTPUT_DIR / "medical"
RUNNER_SCRIPT = PROJECT_ROOT / "run_pipeline.py"

CLASS_NAMES = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion", 4: "Paced / Unknown"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    .material-symbols-rounded, 
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        font-feature-settings: 'liga' 1 !important;
        font-variant-ligatures: discretionary-ligatures !important;
        vertical-align: middle;
    }
    .section-header .material-symbols-rounded {
        color: #6C63FF !important;
        font-size: 1.6rem !important;
    }
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp, [data-testid="stAppViewContainer"] {
        background: #f3f6fb !important;
    }
    [data-testid="stHeader"] {
        background: rgba(243, 246, 251, 0.92) !important;
    }
    h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.02em; color: #1A1A2E; }
    .block-container { max-width: 1200px; padding-top: 2rem; }

    /* ---- hide sidebar collapse button ---- */
    button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1e293b; line-height: 1.1; font-variant-numeric: tabular-nums; }
    .metric-label { font-size: 0.82rem; font-weight: 600; color: #64748b; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .section-header {
        display: flex; align-items: center; gap: 0.5rem;
        margin: 2.5rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #6C63FF20;
    }
    .section-header h2 { margin: 0; font-size: 1.35rem; }

    /* ---- Tabs Redesign ---- */
    div[data-baseweb='tab-list'] {
        margin-left: 0.5rem;
        gap: 0.5rem;
    }
    div[data-baseweb='tab-list'] {
        justify-content: center !important;
    }
    div[data-baseweb='tab-list'] button {
        padding: 0.4rem 0.8rem !important;
        font-size: 14px !important;
        background-color: #e8edf7 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        border: 1px solid #d4dbea !important;
    }
    div[data-baseweb='tab-list'] button:hover {
        background-color: #dfe6f3 !important;
    }
    div[data-baseweb='tab-list'] button[aria-selected='true'] {
        background-color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15,23,42,0.12) !important;
        color: #5b52ff !important;
        font-weight: 600 !important;
        border: 1px solid #b8c2d8 !important;
    }
    div[data-baseweb='tab-highlight'] {
        display: none !important;
    }
    
    /* ---- Global Text Normalization (14px) ---- */
    html, body, [class*="st-"] {
        font-size: 14px !important;
    }
    p, label, li, a, button {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    /* Custom Overrides for Tabs and Selectboxes */
    [data-testid="stSelectbox"] p, 
    [data-testid="stSelectbox"] span, 
    [data-testid="stSelectbox"] div, 
    [data-testid="stTabs"] p, 
    [data-testid="stTabs"] span, 
    [data-testid="stTabs"] div,
    div[data-baseweb="menu"] div,
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] span {
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.2rem !important; }
    h4 { font-size: 1.1rem !important; }
    
    /* ---- Image Shadow ---- */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: none !important;
    }
    .badge {
        display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
    }
    .badge-ok { background: #dcfce7; color: #166534; }
    .badge-warn { background: #fef9c3; color: #854d0e; }

    /* ---- Pipeline Form / Card Shadow ---- */
    [data-testid="stForm"] {
        background: #fff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 28px rgba(15,23,42,0.10), 0 2px 8px rgba(15,23,42,0.06) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stForm"]:hover {
        box-shadow: 0 14px 34px rgba(15,23,42,0.14), 0 4px 12px rgba(15,23,42,0.08) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="baseButton-primary"],
    [data-testid="stBaseButton-primary"] button,
    button[kind="primary"] {
        background: #5b52ff !important;
        border: 1px solid #4338ca !important;
        box-shadow: 0 4px 12px rgba(91,82,255,0.25) !important;
    }
    [data-testid="baseButton-secondary"],
    [data-testid="stBaseButton-secondary"] button,
    button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #b8c2d8 !important;
        color: #1e293b !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.10) !important;
    }

    /* ---- Terminal Widget ---- */
    .term-box {
        background: #1a1a2e;
        border-radius: 16px;
        overflow: hidden;
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25), 0 2px 8px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
        height: auto;
    }
    .term-header {
        background: #3a3a4a;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border-radius: 16px 16px 0 0;
        min-height: 44px;
    }
    .term-cmd {
        flex: 1;
        font-size: 0.78rem;
        color: #c8c8d0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-weight: 500;
    }
    .term-epoch {
        font-size: 0.82rem;
        color: #e0e0e0;
        font-weight: 700;
        white-space: nowrap;
        letter-spacing: 0.02em;
    }
    .term-progress-track {
        height: 4px;
        background: #2a2a3a;
        width: 100%;
    }
    .term-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #22c55e, #4ade80);
        border-radius: 0 2px 2px 0;
        transition: width 0.4s ease;
    }
    .term-body {
        height: 419px;
        padding: 1rem 1.2rem;
        overflow-y: auto;
        overflow-x: hidden;
        font-size: 0.75rem;
        line-height: 1.65;
        color: #d4d4d8;
        white-space: pre-wrap;
        word-break: break-all;
        background: #111119;
        border-radius: 0 0 16px 16px;
        display: flex;
        flex-direction: column-reverse;
    }
    .term-body::-webkit-scrollbar { width: 6px; }
    .term-body::-webkit-scrollbar-track { background: transparent; }
    .term-body::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 3px; }
    .term-empty {
        color: #4a4a5a;
        font-style: italic;
        margin-top: 2rem;
        text-align: center;
        font-size: 0.82rem;
    }

    /* ---- Terminal header row (split for st.columns X button) ---- */
    .term-hdr-bar {
        background: #3a3a4a;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border-radius: 16px 0 0 0;
        min-height: 44px;
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    }
    .term-x-wrap {
        background: #3a3a4a;
        border-radius: 0 16px 0 0;
        min-height: 44px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 0.5rem;
    }
    /* Red X stop button — marker sibling selector */
    .stop-x-marker ~ div [data-testid="baseButton-secondary"],
    .stop-x-marker ~ div [data-testid="stBaseButton-secondary"],
    .stop-x-marker ~ div button {
        width: 28px !important;
        height: 28px !important;
        min-height: 28px !important;
        padding: 0 !important;
        background-color: #ef4444 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: background 0.2s, transform 0.15s !important;
        float: right !important;
    }
    .stop-x-marker ~ div button:hover {
        background-color: #dc2626 !important;
        transform: scale(1.1) !important;
    }
    .stop-x-marker ~ div button p {
        color: #fff !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.9rem !important;
        line-height: 1 !important;
    }
    .term-x-idle {
        width: 28px; height: 28px;
        background: #4a4a5a;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6a6a7a;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .term-body-block {
        margin-top: -1rem;
        border-radius: 0 0 16px 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25), 0 2px 8px rgba(0,0,0,0.15);
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    }

    /* ---- sidebar base ---- */
    section[data-testid="stSidebar"] {
        background: #E9E9E9 !important; /* Pure neutral gray */
        border-right: 1px solid #cccccc !important;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    /* ---- layout logic for sidebar ---- */
    [data-testid="stSidebarContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 100vh !important;
        position: relative !important;
    }
    
    [data-testid="stSidebarContent"]::before {
        content: "ECG XAI";
        display: block;
        position: absolute;
        top: 2rem;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #1e293b; /* Dark text for light sidebar */
        z-index: 999;
    }

    [data-testid="stSidebarUserContent"] {
        position: absolute;
        bottom: 2rem;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        z-index: 999;
    }

    /* ---- sidebar nav items (st.navigation) ---- */
    [data-testid="stSidebarNavSeparator"] {
        display: none !important;
    }
    [data-testid="stSidebarNav"] {
        margin-top: 7rem !important; /* Fixed padding below ECG XAI header so it sits very high up */
        margin-bottom: auto !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebarNav"] ul {
        gap: 2px !important;
        padding: 0 0.5rem !important;
    }
    [data-testid="stSidebarNav"] a {
        color: #475569 !important; /* Slate-600 for light sidebar */
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.25rem 0.75rem !important; /* More compact vertically and horizontally */
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-decoration: none !important;
        position: relative !important;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: rgba(108, 99, 255, 0.08) !important;
        border-color: rgba(108, 99, 255, 0.15) !important;
        color: #6C63FF !important; /* Primary purple */
        transform: translateX(4px);
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(108,99,255,0.1) !important;
        border-color: rgba(108,99,255,0.2) !important;
        color: #6C63FF !important; /* Primary purple selected text */
        font-weight: 600 !important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"]::before {
        content: '' !important;
        position: absolute !important;
        left: 0 !important;
        top: 20% !important;
        height: 60% !important;
        width: 3px !important;
        border-radius: 0 4px 4px 0 !important;
        background: linear-gradient(180deg, #6C63FF, #a78bfa) !important;
    }
    [data-testid="stSidebarNav"] span {
        color: inherit !important;
    }
    /* Hide the default generic header of the nav if any */
    [data-testid="stSidebarNav"] > div:first-child {
        display: none !important;
    }
    div[data-testid="stSidebar"] hr { display: none !important; }

    /* ---- status footer ---- */
    .footer-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    .status-pill {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        padding: 0.4rem 1rem !important;
        border-radius: 999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        background: #ffffff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .status-pill.ok {
        border: 1px solid #86efac !important;
        color: #166534 !important;
    }
    .status-pill.err {
        border: 1px solid #fca5a5 !important;
        color: #991b1b !important;
    }
    .status-dot {
        width: 8px !important; height: 8px !important; border-radius: 50% !important;
        display: inline-block !important;
    }
    .status-dot.ok { background: #22c55e !important; box-shadow: 0 0 8px rgba(34,197,94,0.6) !important; }
    .status-dot.err { background: #ef4444 !important; box-shadow: 0 0 8px rgba(239,68,68,0.6) !important; }
    .sidebar-version {
        font-size: 0.62rem;
        color: #3a3e52 !important;
        margin-top: 0.4rem;
        letter-spacing: 0.06em;
    }



    .log-area {
        background: #1e1e2e; color: #cdd6f4; border-radius: 10px;
        padding: 1rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; max-height: 420px; overflow-y: auto;
        white-space: pre-wrap; line-height: 1.55;
    }

    /* ---- styled report tables ---- */
    .report-table { width: 100%; border-collapse: separate; border-spacing: 0;
        border-radius: 12px; overflow: hidden; border: 1px solid #b8c2d8;
        font-size: 0.88rem; margin: 0.5rem 0; background:#ffffff; }
    .report-table thead th {
        background: #e2e8f0; color: #1e293b; font-weight: 600;
        padding: 0.65rem 1rem; text-align: left; border-bottom: 2px solid #94a3b8;
    }
    .report-table tbody td {
        padding: 0.55rem 1rem; color: #334155; border-bottom: 1px solid #cbd5e1;
    }
    .report-table tbody tr:last-child td { border-bottom: none; }
    .report-table tbody tr:hover { background: #f9fafb; }
    .report-table .row-label { font-weight: 600; color: #1a1a2e; }
    .report-table .num { font-variant-numeric: tabular-nums; text-align: right; }

    .kv-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;
        margin: 0.5rem 0;
    }
    .kv-item {
        background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
        padding: 0.7rem 1rem;
        transition: background 0.2s;
        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
    }
    .kv-item:hover { background: #f0f2ff; }
    .kv-item .kv-key { font-size: 0.75rem; color: #6b7280; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.04em; }
    .kv-item .kv-val { font-size: 1.05rem; color: #1a1a2e; font-weight: 600; margin-top: 0.15rem; }
    </style>
    """, unsafe_allow_html=True)


def _metric_card(label: str, value: str):
    st.markdown(f"""
    <div style="padding: 0.8rem 0;">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def _section(icon: str, title: str):
    if icon.isascii():
        icon_html = f'<span class="material-symbols-rounded">{icon}</span>'
    else:
        icon_html = f'<span style="font-size:1.4rem">{icon}</span>'
    st.markdown(f'<div class="section-header">{icon_html}<h2>{title}</h2></div>', unsafe_allow_html=True)


def _render_classification_report(text: str):
    """Parse sklearn classification_report text into a styled HTML table."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    # Header row
    html = '<table class="report-table"><thead><tr>'
    html += '<th>Class</th><th class="num">Precision</th><th class="num">Recall</th>'
    html += '<th class="num">F1-Score</th><th class="num">Support</th></tr></thead><tbody>'
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        # Try to detect data rows: last 4 tokens should be numbers
        nums = []
        for p in parts[-4:]:
            try:
                float(p)
                nums.append(p)
            except ValueError:
                break
        if len(nums) == 4:
            label = " ".join(parts[:-4]).strip()
            if not label:
                continue
            html += f'<tr><td class="row-label">{label}</td>'
            for n in nums:
                html += f'<td class="num">{n}</td>'
            html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def _render_kv_report(text: str):
    """Render a key=value text file as a styled grid of cards."""
    items = []
    for line in text.strip().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            label = k.strip().replace('_', ' ').title()
            items.append((label, v.strip()))
    if not items:
        st.text(text)
        return
    html = '<div class="kv-grid">'
    for label, val in items:
        html += f'<div class="kv-item"><div class="kv-key">{label}</div><div class="kv-val">{val}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _render_plain_report(text: str):
    """Render multi-line plain text as a clean info-box with proper lines."""
    lines_html = '<br>'.join(line if line.strip() else '' for line in text.splitlines())
    st.markdown(f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;'
                f'padding:1rem 1.2rem;font-size:0.88rem;color:#374151;line-height:1.7;">'
                f'{lines_html}</div>', unsafe_allow_html=True)


def _read_text(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _alignment_column(df: pd.DataFrame) -> str | None:
    if "alignment_score" in df.columns:
        return "alignment_score"
    if "dice" in df.columns:
        return "dice"
    return None


def _sample_alignment_text(sample_rows: pd.DataFrame) -> str:
    metric = _alignment_column(sample_rows)
    if metric is None:
        return "N/A"
    values = pd.to_numeric(sample_rows[metric], errors="coerce").dropna()
    if values.empty:
        return "N/A"
    return f"{values.median():.2f}"


def _list_images(folder: Path, prefix: str = "") -> list[Path]:
    if not folder.exists():
        return []
    imgs = sorted(folder.glob("*.png"))
    if prefix:
        imgs = [p for p in imgs if p.name.startswith(prefix)]
    return imgs


def _sample_ids_from_xai() -> list[int]:
    """Return sample IDs that have both a SHAP image and a CSV row.

    Images from earlier pipeline runs accumulate in output/xai/ and may not
    have a matching row in xai_clinical_comparison.csv (which is rewritten on
    every run).  We intersect both sources so the picker only shows samples
    that the app can fully display.
    """
    imgs = _list_images(XAI_DIR, "shap_sample_")
    img_ids: set[int] = set()
    for p in imgs:
        try:
            img_ids.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass

    # If the CSV exists, restrict to IDs that are present in it
    csv_path = REPORTS_DIR / "xai_clinical_comparison.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["sample_idx"])
            csv_ids = set(df["sample_idx"].dropna().astype(int).tolist())
            img_ids = img_ids & csv_ids
        except Exception:
            pass  # fall back to image-only list

    return sorted(img_ids)


def _render_plot_image(path: Path, missing_text: str, width: bool = True) -> None:
    if path.exists():
        st.image(Image.open(path), use_container_width=width)
    else:
        st.caption(missing_text)



# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    # ── Hero architecture diagram ──────────────────────────────────────────
    hero_html = (Path(__file__).parent / "overview_hero.html").read_text(encoding="utf-8")
    st.markdown(hero_html, unsafe_allow_html=True)



def _parse_epoch_info(logs: list[str]) -> dict:
    """Extract epoch progress, acc, loss, and epoch duration from Keras log lines."""
    import re
    info = {"cur": 0, "total": 0, "acc": "", "loss": "", "time": ""}
    for line in reversed(logs):
        # Match 'Epoch 5/30'
        if info["cur"] == 0:
            m = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            if m:
                info["cur"], info["total"] = int(m.group(1)), int(m.group(2))
        # Match Keras end-of-epoch line with metrics
        # e.g. "187/187 ━━━ 3s 18ms/step - accuracy: 0.9534 - loss: 0.1892 - val_accuracy: ..."
        if not info["acc"]:
            m_acc = re.search(r'accuracy:\s*([0-9.]+)', line)
            m_loss = re.search(r'(?<!val_)loss:\s*([0-9.]+)', line)
            if m_acc and m_loss:
                info["acc"] = m_acc.group(1)
                info["loss"] = m_loss.group(1)
        # Match epoch time: "3s", "12s", "45s" at start of metrics line
        if not info["time"]:
            m_time = re.search(r'\b(\d+)s\s+\d+ms/step', line)
            if m_time:
                info["time"] = m_time.group(1) + "s"
        if info["cur"] and info["acc"] and info["time"]:
            break
    return info


def _render_terminal(logs: list[str], running: bool,
                     epoch_info: dict, total_epochs_cfg: int):
    """Build the HTML terminal widget."""
    import html as html_mod
    epoch_cur = epoch_info["cur"]
    epoch_total = epoch_info["total"]
    denom = epoch_total if epoch_total > 0 else total_epochs_cfg
    pct = min(100, int((epoch_cur / denom) * 100)) if denom > 0 and epoch_cur > 0 else 0

    # Left info: acc/loss + epoch time
    if epoch_info["acc"] and epoch_info["loss"]:
        left_label = f'acc/loss: {epoch_info["acc"]}/{epoch_info["loss"]}'
        if epoch_info["time"]:
            left_label += f'  ·  {epoch_info["time"]}/epoch'
    elif running:
        left_label = "waiting for first epoch…"
    else:
        left_label = "idle"

    # Right info: epoch counter
    if epoch_cur > 0:
        epoch_label = f"epoch {epoch_cur}/{epoch_total if epoch_total else denom}"
    elif running:
        epoch_label = "starting…"
    else:
        epoch_label = ""

    # No close button in HTML — the real st.button is overlaid via CSS

    # Body — wrap in a div for column-reverse to auto-scroll
    if logs:
        body_text = html_mod.escape("\n".join(logs[-200:]))
        body_inner = f'<div>{body_text}</div>'
    else:
        body_inner = '<div><span class="term-empty">Terminal output will appear here…</span></div>'

    bar_bg = "linear-gradient(90deg, #22c55e, #4ade80)" if (running or epoch_cur > 0) else "#3a3a4a"

    left_esc = html_mod.escape(left_label)
    epoch_esc = html_mod.escape(epoch_label)

    return f"""<div class="term-box">
<div class="term-header">
<span class="term-cmd">{left_esc}</span>
<span class="term-epoch">{epoch_esc}</span>
</div>
<div class="term-progress-track"><div class="term-progress-bar" style="width:{pct}%;background:{bar_bg}"></div></div>
<div class="term-body">{body_inner}</div>
</div>"""


def page_run_pipeline():
    st.markdown("## :material/play_arrow: Run Pipeline")
    st.caption("Configure and launch `run_pipeline.py` directly from the dashboard.")

    # -- Initialise session state for the persistent terminal --
    for key, default in [("pipe_proc", None), ("pipe_thread", None),
                         ("pipe_logs", []), ("pipe_cmd", ""),
                         ("pipe_running", False), ("pipe_epochs_cfg", 20),
                         ("pipe_done_code", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_left, col_right = st.columns([2, 3], gap="large")

    # ── LEFT: Configuration form ─────────────────────────────────────────
    # Defaults sourced from PipelineConfig in config.py:
    #   epochs=20, batch_size=64, samples_per_class=20,
    #   shap_background_size=200, mode="full", run_xai=True
    with col_left:
        with st.form("run_form"):
            mode = st.selectbox("Mode", ["full", "resume_post_xai"],
                                help="full = train + evaluate + XAI; resume = skip training")
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20)
            batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=64, step=8)
            samples_per_class = st.number_input("Samples per Class (XAI)",
                                                min_value=1, max_value=50, value=20)
            shap_bg = st.number_input("SHAP Background Size",
                                      min_value=50, max_value=2000, value=200, step=50)
            no_xai = st.checkbox("Skip XAI (--no-xai)", value=False)

            submitted = st.form_submit_button("🚀  Launch Pipeline",
                                              type="primary", use_container_width=True)

    # ── Handle launch ────────────────────────────────────────────────────
    if submitted and not st.session_state.pipe_running:
        cmd = [sys.executable, str(RUNNER_SCRIPT), "--mode", mode,
               "--epochs", str(epochs), "--batch-size", str(batch_size),
               "--samples-per-class", str(samples_per_class),
               "--shap-background-size", str(shap_bg)]
        if no_xai:
            cmd.append("--no-xai")

        st.session_state.pipe_cmd = " ".join(cmd)
        st.session_state.pipe_logs = []
        st.session_state.pipe_running = True
        st.session_state.pipe_epochs_cfg = epochs
        st.session_state.pipe_done_code = None

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(PROJECT_ROOT), bufsize=1)
        st.session_state.pipe_proc = proc

        def _stream(p, log_list):
            try:
                for line in iter(p.stdout.readline, ""):
                    log_list.append(line.rstrip())
            except Exception:
                pass
            p.wait()

        t = threading.Thread(target=_stream,
                             args=(proc, st.session_state.pipe_logs), daemon=True)
        t.start()
        st.session_state.pipe_thread = t

    # ── Check if process finished naturally ───────────────────────────────
    if st.session_state.pipe_running:
        proc = st.session_state.pipe_proc
        if proc and proc.poll() is not None:
            st.session_state.pipe_running = False
            st.session_state.pipe_done_code = proc.returncode

    # ── RIGHT: Terminal widget with stop X overlaid ─────────────────────────
    with col_right:
        logs = st.session_state.pipe_logs
        running = st.session_state.pipe_running
        epoch_info = _parse_epoch_info(logs)

        # Wrapper div for the terminal; the stop button below uses CSS sibling selectors
        st.markdown('<div class="term-wrapper">', unsafe_allow_html=True)

        term_html = _render_terminal(
            logs, running, epoch_info, st.session_state.pipe_epochs_cfg
        )
        st.markdown(term_html, unsafe_allow_html=True)

        # Render the stop X button — marker + sibling CSS makes it red
        if running:
            st.markdown('<div class="stop-x-marker"></div>', unsafe_allow_html=True)
            if st.button("✕", key="stop_pipe"):
                proc = st.session_state.pipe_proc
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                st.session_state.pipe_running = False
                st.session_state.pipe_done_code = -1
                st.session_state.pipe_logs.append("\n── Process terminated by user ──")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Completion status
        done_code = st.session_state.pipe_done_code
        if done_code is not None and not running:
            if done_code == 0:
                st.success("✅  Pipeline finished successfully!")
            elif done_code == -1:
                st.warning("⚠️  Pipeline was stopped by user.")
            else:
                st.error(f"❌  Pipeline exited with code {done_code}")

    # ── Auto-refresh while running ───────────────────────────────────────
    if st.session_state.pipe_running:
        time.sleep(1.2)
        st.rerun()




def page_xai_explorer():
    st.markdown("<div style='margin-top: -1rem;'></div>", unsafe_allow_html=True)
    sample_ids = _sample_ids_from_xai()
    if not sample_ids:
        st.info("No XAI sample images found. Run the pipeline with XAI enabled.")
        return

    xai_df = _load_csv(REPORTS_DIR / "xai_clinical_comparison.csv")

    col_id, col_card = st.columns([1, 2.5])
    
    with col_id:
        chosen_id = st.selectbox("Sample ID", sample_ids)
        
    with col_card:
        if xai_df is not None:
            row = xai_df[xai_df["sample_idx"] == chosen_id]
            if not row.empty:
                first = row.iloc[0]
                alignment_text = _sample_alignment_text(row)
                is_correct = first['y_true'] == first['y_pred']
                status_text = 'Correct' if is_correct else 'Misclassified'
                status_icon = '✅' if is_correct else '🔴'
                badge_bg = '#dcfce7' if is_correct else '#fee2e2'
                badge_color = '#166534' if is_correct else '#dc2626'

                st.markdown(f"""
                <div style="font-size:14px; color:#1A1A2E; margin-bottom:0.31rem; font-weight:normal; text-align:center;">Classification</div>
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:0.8rem 1rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:0.8rem; flex-wrap:wrap; margin-bottom:0.6rem;">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="color:#64748b; font-size:14px; font-weight:500;">Status:</span>
                            <span style="background:{badge_bg}; color:{badge_color}; padding:0.15rem 0.45rem; border-radius:4px; font-size:14px; font-weight:600; display:inline-flex; align-items:center; gap:0.2rem;">{status_icon} {status_text}</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="color:#64748b; font-size:14px; font-weight:500;">Confidence:</span>
                            <span style="font-size:14px; font-weight:600; color:#1e293b;">{first['pred_confidence']:.2f}</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="color:#64748b; font-size:14px; font-weight:500;">Alignment:</span>
                            <span style="font-size:14px; font-weight:600; color:#1e293b;">{alignment_text}</span>
                        </div>
                    </div>
                    <div style="border-top:1px solid #f1f5f9; padding-top:0.6rem; display:flex; align-items:center; gap:1.5rem; font-size:14px;">
                        <div><span style="color:#64748b; font-weight:500;">True Class:</span> <span style="font-weight:600; color:#1e293b;">{CLASS_NAMES.get(int(first['y_true']), first['y_true'])}</span></div>
                        <div style="color:#cbd5e1;">|</div>
                        <div><span style="color:#64748b; font-weight:500;">Predicted Class:</span> <span style="font-weight:600; color:#1e293b;">{CLASS_NAMES.get(int(first['y_pred']), first['y_pred'])}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Stale image from an older pipeline run — no CSV row available
                st.markdown("""
                <div style="font-size:14px; color:#1A1A2E; margin-bottom:0.31rem; font-weight:normal; text-align:center;">Classification</div>
                <div style="background:#fff8f0; border:1px solid #fed7aa; border-radius:8px; padding:0.8rem 1rem;
                            box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);">
                    <span style="color:#92400e; font-size:13.5px; font-weight:500;">
                    ⚠️ No classification data — this image is from a previous pipeline run.
                    Re-run the pipeline to refresh all samples.
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Classification data not yet available. Run the pipeline first.")

    # ── Ontology inference card ──────────────────────────────────────────
    rapport_path = XAI_DIR / f"rapport_sample_{chosen_id}.txt"
    if rapport_path.exists():
        rapport_raw = rapport_path.read_text(encoding="utf-8")
        onto_diag = ""
        onto_explanation = ""
        for line in rapport_raw.splitlines():
            if "Ontological Diagnosis" in line and ":" in line:
                onto_diag = line.split(":", 1)[-1].strip()
            elif line.startswith("Explanation :"):
                onto_explanation = line.split(":", 1)[-1].strip()

        diag_colors = {
            "Normal Beat": ("#dcfce7", "#166534"),
            "Ventricular Ectopic Beat": ("#fee2e2", "#dc2626"),
            "Supraventricular Ectopic Beat": ("#fef3c7", "#92400e"),
            "Fusion Beat": ("#ede9fe", "#5b21b6"),
            "Paced / Unknown Beat": ("#e0f2fe", "#0369a1"),
        }
        diag_bg, diag_fg = diag_colors.get(onto_diag, ("#f1f5f9", "#334155"))

        st.markdown(f"""
        <div style="font-size:14px; color:#1A1A2E; margin-bottom:0.31rem; margin-top:0.6rem; font-weight:normal; text-align:center;">Ontology Inference</div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:0.8rem 1rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.6rem;">
                <span style="color:#64748b; font-size:14px; font-weight:500;">Diagnosis:</span>
                <span style="background:{diag_bg}; color:{diag_fg}; padding:0.15rem 0.55rem; border-radius:4px; font-size:14px; font-weight:600;">{onto_diag}</span>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding-top:0.6rem;">
                <p style="color:#334155; font-size:13.5px; line-height:1.65; margin:0; text-align:justify;">{onto_explanation}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: -1.5rem; margin-bottom: -1.5rem;'><hr style='margin: 0; border: none; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)
    view_col, tab_col = st.columns([1, 2.5])
    
    with view_col:
        view_mode = st.selectbox(
            "Plot view",
            [
                "XAI Method attention overlay",
                "XAI Method comparison",
                "Clinical Alignment Validation",
                "Model Performance",
            ],
        )

    with tab_col:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        
        if view_mode == "XAI Method attention overlay":
            methods = ["shap", "gradcam", "integratedgradients"]
            tabs = st.tabs(["SHAP", "GradCAM", "Integrated Gradients"])
            for tab, method in zip(tabs, methods):
                with tab:
                    p = XAI_DIR / f"{method}_sample_{chosen_id}.png"
                    if p.exists():
                        st.image(Image.open(p), use_container_width=True)
                    else:
                        st.caption(f"No {method} image for sample {chosen_id}.")

        elif view_mode == "XAI Method comparison":
            tabs = st.tabs(["Sample Comparison", "Alignment Summary", "Agreement Heatmap"])
            
            with tabs[0]:
                p = XAI_DIR / f"method_comparison_sample_{chosen_id}.png"
                if p.exists():
                    _, center, _ = st.columns([0.05, 0.9, 0.05])
                    center.image(Image.open(p), use_container_width=True)
                else:
                    st.caption("Comparison image not found.")
            
            with tabs[1]:
                p = FIGURES_DIR / "xai_method_metric_summary.png"
                if p.exists():
                    st.image(Image.open(p), use_container_width=True)
                else:
                    st.caption("Metric summary not generated yet.")
                    
            with tabs[2]:
                p = FIGURES_DIR / "xai_inter_method_agreement.png"
                if p.exists():
                    _, center, _ = st.columns([0.15, 0.7, 0.15])
                    center.image(Image.open(p), use_container_width=True)
                else:
                    st.caption("Agreement heatmap not generated yet.")

        elif view_mode == "Clinical Alignment Validation":
            p = FIGURES_DIR / "clinical_occlusion_comparison.png"
            if p.exists():
                _, center, _ = st.columns([0.12, 0.76, 0.12])
                center.image(Image.open(p), use_container_width=True)
            else:
                st.caption("Clinical alignment validation plot not generated yet.")

        elif view_mode == "Model Performance":
            tabs = st.tabs(["Confusion Matrix", "Training Loss", "Training Accuracy"])
            with tabs[0]:
                _render_plot_image(
                    FIGURES_DIR / "confusion_matrix.png",
                    "Confusion matrix not generated yet.",
                )
            with tabs[1]:
                _render_plot_image(
                    FIGURES_DIR / "training_loss.png",
                    "Training loss plot will appear after the next full training run.",
                )
            with tabs[2]:
                _render_plot_image(
                    FIGURES_DIR / "training_accuracy.png",
                    "Training accuracy plot will appear after the next full training run.",
                )



def page_reports():
    st.markdown("## :material/description: Reports")

    report_files = {
        "Classification Report": REPORTS_DIR / "classification_report.txt",
        "Experiment Summary": REPORTS_DIR / "experiment_summary.txt",
        "Ontology Status": REPORTS_DIR / "ontology_status.txt",
    }
    csv_files = {
        "XAI Alignment Summary": REPORTS_DIR / "xai_clinical_comparison.csv",
        "XAI Inter-Method Agreement": REPORTS_DIR / "xai_inter_method_agreement.csv",
        "Ontology Component Durations": REPORTS_DIR / "ontology_component_durations.csv",
    }

    # --- Text Reports as tabs ---
    available_reports = [(name, path) for name, path in report_files.items() if path.exists()]
    if available_reports:
        _section("edit_document", "Text Reports")
        tabs = st.tabs([name for name, _ in available_reports])
        for tab, (name, path) in zip(tabs, available_reports):
            with tab:
                content = _read_text(path)
                if content:
                    if name == "Classification Report":
                        _render_classification_report(content)
                    elif '=' in content.splitlines()[0]:
                        _render_kv_report(content)
                    else:
                        _render_plain_report(content)

    # --- CSV Data Tables as tabs ---
    available_csvs = [(name, path) for name, path in csv_files.items() if path.exists()]
    if available_csvs:
        _section("📊", "Data Tables")
        tabs = st.tabs([name for name, _ in available_csvs])
        for tab, (name, path) in zip(tabs, available_csvs):
            with tab:
                df = _load_csv(path)
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ Download {name}", csv_data, file_name=path.name, mime="text/csv")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ECG XAI Pipeline", page_icon="💓", layout="wide",
                       initial_sidebar_state="expanded")
    _css()

    pg = st.navigation([
        st.Page(page_overview, title="Overview", icon=":material/home:"),
        st.Page(page_run_pipeline, title="Run Pipeline", icon=":material/play_arrow:"),

        st.Page(page_xai_explorer, title="XAI Explorer", icon=":material/science:"),
        st.Page(page_reports, title="Reports", icon=":material/description:"),
    ])

    with st.sidebar:
        # -- status footer --
        has_output = OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())
        s_class = "ok" if has_output else "err"
        s_text = "Outputs ready" if has_output else "No outputs"
        st.markdown(f"""
        <div class="footer-wrapper">
            <div class="status-pill {s_class}">
                <span class="status-dot {s_class}"></span>
                {s_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    pg.run()


if __name__ == "__main__":
    main()

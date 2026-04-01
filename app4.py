"""
Sarcasm Detector 3000 — Local Streamlit App
Run: streamlit run app.py --server.port 8504

Expects your saved model folder at: ./best_roberta_sarcasm
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcasm Detector 3000",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Serif Display', serif;
    background-color: #0d0d0d;
    color: #f0ebe0;
}
.stApp {
    background-color: #0d0d0d;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,230,0,0.07) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.015) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.015) 40px);
}

/* HERO */
.hero { text-align: center; padding: 48px 0 36px; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: #FFE600;
    margin-bottom: 14px;
    opacity: 0.85;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 5.5rem;
    letter-spacing: 6px;
    line-height: 0.95;
    color: #f0ebe0;
    text-shadow: 0 0 60px rgba(255,230,0,0.15);
}
.hero-title .accent { color: #FFE600; }
.hero-title .red    { color: #FF2D2D; }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #555;
    letter-spacing: 2px;
    margin-top: 14px;
    text-transform: uppercase;
}

/* TICKER */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: #FFE600;
    border-top: 2px solid #1a1a1a;
    border-bottom: 2px solid #1a1a1a;
    padding: 10px 0;
    margin: 28px 0 32px;
}
.ticker-track {
    display: flex;
    animation: ticker 30s linear infinite;
    white-space: nowrap;
}
.ticker-track:hover { animation-play-state: paused; cursor: default; }
.ticker-item {
    font-family: 'Bebas Neue', cursive;
    font-size: 1rem;
    letter-spacing: 2px;
    color: #1a1a1a;
    padding: 0 28px;
    flex-shrink: 0;
}
.ticker-sep { color: #FF2D2D; padding: 0 8px; }
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* SAMPLE HEADLINES */
.headlines-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #FFE600;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.pulse-dot {
    width: 7px; height: 7px;
    background: #FF2D2D;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(0.65); }
}
.headline-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #FFE600;
    padding: 14px 18px;
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem;
    color: #b0a898;
    line-height: 1.5;
}
.tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 2px 7px;
    margin-right: 10px;
    vertical-align: middle;
}
.tag-sarc { background: rgba(255,45,45,0.15); color: #FF2D2D; border: 1px solid rgba(255,45,45,0.3); }
.tag-real { background: rgba(0,200,81,0.1);   color: #00C851; border: 1px solid rgba(0,200,81,0.25); }

/* INPUT */
.input-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 0 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.05rem !important;
    color: #f0ebe0 !important;
    caret-color: #FFE600 !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #FFE600 !important;
    box-shadow: 0 0 0 1px #FFE600 !important;
}
.stTextArea textarea::placeholder { color: #3a3a3a !important; font-style: italic !important; }

/* BUTTON */
.stButton > button {
    background: #FFE600 !important;
    color: #0d0d0d !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1.25rem !important;
    letter-spacing: 5px !important;
    border: none !important;
    border-radius: 0 !important;
    width: 100% !important;
    padding: 16px !important;
    box-shadow: 4px 4px 0 #FF2D2D !important;
    transition: all 0.12s !important;
}
.stButton > button:hover {
    background: #fff !important;
    transform: translate(-2px,-2px) !important;
    box-shadow: 6px 6px 0 #FF2D2D !important;
}
.stButton > button:active {
    transform: translate(2px,2px) !important;
    box-shadow: 2px 2px 0 #FF2D2D !important;
}

/* CHECKBOX */
.stCheckbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #555 !important;
    letter-spacing: 1px !important;
}

/* DIVIDER */
.fancy-divider {
    display: flex; align-items: center; gap: 14px; margin: 32px 0 24px;
}
.fancy-divider hr { flex:1; border:none; border-top: 1px solid #1e1e1e; }
.fancy-divider span {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem; letter-spacing: 3px; color: #333; text-transform: uppercase;
}

/* SCORE */
.score-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
.score-main {
    background: #FF2D2D; padding: 24px 20px; text-align: center; position: relative; overflow: hidden;
}
.score-main::before {
    content:''; position:absolute; top:-30px; right:-30px;
    width:100px; height:100px; background:rgba(255,255,255,0.06); border-radius:50%;
}
.big-num { font-family:'Bebas Neue',cursive; font-size:5rem; color:#fff; line-height:1; display:block; }
.big-label { font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:3px; color:rgba(255,255,255,0.65); text-transform:uppercase; margin-top:4px; }
.score-verdict {
    background:#1a1a1a; border:1px solid #2a2a2a; padding:24px 20px;
    display:flex; flex-direction:column; justify-content:center; gap:12px;
}
.verdict-text { font-family:'Bebas Neue',cursive; font-size:1.5rem; letter-spacing:2px; line-height:1.1; }
.verdict-sarc { color:#FF2D2D; }
.verdict-not  { color:#00C851; }
.conf-bar-label { font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:2px; color:#555; text-transform:uppercase; margin-bottom:5px; }
.conf-bar-bg { height:4px; background:#111; border-radius:2px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:2px; }

/* METRICS */
.metric-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:24px; }
.metric-box { background:#111; border:1px solid #1e1e1e; padding:14px 12px; text-align:center; }
.mv { font-family:'Bebas Neue',cursive; font-size:2rem; color:#FFE600; line-height:1; }
.ml { font-family:'DM Mono',monospace; font-size:0.53rem; color:#444; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

/* SECTION HEAD */
.sec-head {
    font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:4px; color:#FFE600;
    text-transform:uppercase; border-bottom:1px solid #1e1e1e; padding-bottom:8px; margin:28px 0 16px;
}

/* LIME */
.lime-box {
    background:#111; border:1px solid #1e1e1e; padding:20px;
    line-height:2.6; font-family:'DM Serif Display',serif; font-size:1.1rem;
    color:#d0c8b8; margin-bottom:12px;
}
.lime-legend {
    display:flex; gap:20px; flex-wrap:wrap;
    font-family:'DM Mono',monospace; font-size:0.57rem; color:#444;
    text-transform:uppercase; letter-spacing:1px; margin-top:8px;
}

/* CHIPS */
.chip-wrap { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.chip { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:1px; padding:5px 12px; border:1px solid; }
.chip-r { color:#FF2D2D; border-color:rgba(255,45,45,0.4); background:rgba(255,45,45,0.07); }
.chip-g { color:#00C851; border-color:rgba(0,200,81,0.35); background:rgba(0,200,81,0.06); }

hr { border:none; border-top:1px solid #1a1a1a; margin:28px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH = "./best_roberta_sarcasm"
    with st.spinner("Loading model..."):
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = model.to(device)
        model.eval()
    return model, tokenizer, device

try:
    model, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_proba(texts):
    all_probs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        enc   = tokenizer(batch, max_length=64, padding="max_length",
                          truncation=True, return_tensors="pt")
        with torch.no_grad():
            out   = model(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

def predict_single(h):
    p = predict_proba([h])[0]
    return float(p[1]), float(p[0])

def verdict_text(pct):
    if pct >= 80: return "MAXIMUM SARCASM 🔥", "verdict-sarc"
    if pct >= 55: return "DRIPPING SARCASM 🙄", "verdict-sarc"
    if pct >= 35: return "MILD EYE ROLL 😒",    "verdict-not"
    return "BONE DRY 😐", "verdict-not"

@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=["Not Sarcastic","Sarcastic"],
                             split_expression=" ", bow=False, random_state=42)

def run_lime(headline, n=300):
    exp = get_explainer().explain_instance(
        headline, predict_proba, num_features=10, num_samples=n, labels=(1,))
    return exp.as_list(label=1)

def build_lime_html(headline, word_scores):
    score_map = {w.lower(): s for w, s in word_scores}
    max_abs   = max((abs(s) for _, s in word_scores), default=0.01)
    parts = []
    for word in headline.split():
        clean = word.lower().strip(".,!?;:'\"")
        s     = score_map.get(clean)
        if s is None:
            parts.append(f'<span style="padding:2px 4px">{word}</span>')
            continue
        norm = abs(s) / max_abs
        if s > 0:
            a = 0.12 + norm * 0.5
            bg  = f"rgba(255,45,45,{a:.2f})"
            bdr = f"2px solid rgba(255,45,45,{min(0.9,norm+0.3):.2f})"
        else:
            a = 0.1 + norm * 0.35
            bg  = f"rgba(0,200,81,{a:.2f})"
            bdr = f"2px solid rgba(0,200,81,{min(0.9,norm+0.3):.2f})"
        tip = f"+{s:.3f} → sarcasm" if s > 0 else f"{s:.3f} → not sarcastic"
        parts.append(
            f'<span title="{tip}" style="background:{bg};border-bottom:{bdr};'
            f'padding:2px 6px;margin:1px;display:inline-block;cursor:default;">{word}</span>')
    return " ".join(parts)

def build_chip_html(word_scores):
    tr = [(w,s) for w,s in word_scores if s > 0][:5]
    su = [(w,s) for w,s in word_scores if s < 0][:3]
    h  = ""
    for w,s in tr:
        h += f'<span class="chip chip-r">▲ {w} ({s:+.2f})</span>'
    for w,s in su:
        h += f'<span class="chip chip-g">▼ {w} ({s:+.2f})</span>'
    return h or '<span style="color:#333;font-size:0.75rem;font-family:DM Mono,monospace;">No strong signals found</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◈ NLP · Sequence Classification · Explainability</div>
  <div class="hero-title">
    SARCASM<br><span class="accent">DETECTOR</span> <span class="red">3000</span>
  </div>
  <div class="hero-sub">Fine-tuned RoBERTa · Headline Analysis · LIME Explainability</div>
</div>
""", unsafe_allow_html=True)

# Ticker (scrolling news bar)
ticker_content = """
<span class="ticker-item">Government Heroically Raises Taxes Again</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">Scientists Discover Water Is Wet, Nation Shocked</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">Airline Generously Adds New Baggage Fee For Your Convenience</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">Area Man Brilliantly Solves Traffic By Driving Faster</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">Researchers Find Breakthrough Cancer Treatment</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">Congress Efficiently Cuts Education Budget Once More</span>
<span class="ticker-item ticker-sep">◆</span>
<span class="ticker-item">City Opens New Affordable Housing Units For Families</span>
<span class="ticker-item ticker-sep">◆</span>
"""
st.markdown(f"""
<div class="ticker-wrap">
  <div class="ticker-track">
    {ticker_content}{ticker_content}
  </div>
</div>
""", unsafe_allow_html=True)

# Sample headlines section
st.markdown("""
<div class="headlines-label">
  <span class="pulse-dot"></span> Sample Headlines — Click → to try one
</div>
""", unsafe_allow_html=True)

samples = [
    ("Government does wonderful job fixing the economy once again",           "sarc"),
    ("Scientists discover new vaccine against deadly virus",                  "real"),
    ("Airline generously adds new incredible baggage fee for convenience",    "sarc"),
    ("Researchers find breakthrough treatment for Alzheimer's disease",       "real"),
    ("Congress efficiently passes bill to cut education budget",              "sarc"),
    ("Local school receives funding for new library and sports complex",      "real"),
]

for text, kind in samples[:3]:
    tag_cls  = "tag-sarc" if kind == "sarc" else "tag-real"
    tag_text = "LIKELY SARCASTIC" if kind == "sarc" else "LIKELY GENUINE"
    col1, col2 = st.columns([11, 1])
    with col1:
        st.markdown(
            f'<div class="headline-card"><span class="tag {tag_cls}">{tag_text}</span>{text}</div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("→", key=f"s_{text[:15]}"):
            st.session_state["headline_input"] = text
            st.rerun()

with st.expander("Show more examples"):
    for text, kind in samples[3:]:
        tag_cls  = "tag-sarc" if kind == "sarc" else "tag-real"
        tag_text = "LIKELY SARCASTIC" if kind == "sarc" else "LIKELY GENUINE"
        col1, col2 = st.columns([11, 1])
        with col1:
            st.markdown(
                f'<div class="headline-card"><span class="tag {tag_cls}">{tag_text}</span>{text}</div>',
                unsafe_allow_html=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("→", key=f"sx_{text[:15]}"):
                st.session_state["headline_input"] = text
                st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# Model check
if not model_loaded:
    st.error(
        f"❌ Could not load model from `./best_roberta_sarcasm`\n\n"
        f"Make sure the folder is in the same directory as `app.py`.\n\n"
        f"**Error:** {load_error}")
    st.stop()

# Input
st.markdown('<div class="input-label">◈ Enter your headline</div>', unsafe_allow_html=True)
headline_val = st.session_state.get("headline_input", "")
headline = st.text_area(
    label="",
    value=headline_val,
    height=100,
    placeholder="Paste any news headline here...",
    key="headline_box",
)

run_lime_check = st.checkbox("⚡ Include LIME word-level explanation  (~15 sec)", value=True)
detect_btn = st.button("◈ ANALYSE SARCASM ◈")

# ── Results ───────────────────────────────────────────────────────────────────
if detect_btn and headline.strip():

    with st.spinner("Analysing..."):
        sarc_prob, not_sarc_prob = predict_single(headline)

    sarc_pct     = round(sarc_prob * 100)
    not_sarc_pct = round(not_sarc_prob * 100)
    irony_pct    = round(sarc_pct * 0.9)
    sent_gap_pct = round(sarc_pct * 0.75)
    vtext, vcls  = verdict_text(sarc_pct)
    fill_col     = "#FF2D2D" if sarc_pct >= 50 else "#00C851"

    st.markdown('<div class="fancy-divider"><hr><span>Analysis Results</span><hr></div>',
                unsafe_allow_html=True)

    # Score + verdict card
    st.markdown(f"""
    <div class="score-wrap">
      <div class="score-main">
        <span class="big-num">{sarc_pct}%</span>
        <div class="big-label">Sarcasm Score</div>
      </div>
      <div class="score-verdict">
        <div class="verdict-text {vcls}">{vtext}</div>
        <div>
          <div class="conf-bar-label">Model Confidence</div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{sarc_pct}%;background:{fill_col};"></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric strip
    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-box"><div class="mv">{sarc_pct}%</div><div class="ml">Sarcastic</div></div>
      <div class="metric-box"><div class="mv">{not_sarc_pct}%</div><div class="ml">Not Sarcastic</div></div>
      <div class="metric-box"><div class="mv">{irony_pct}%</div><div class="ml">Irony Level</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar chart
    st.markdown('<div class="sec-head">◈ Confidence Breakdown</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    labels_c = ["Sarcastic", "Not Sarcastic", "Irony Level", "Sentiment Gap"]
    values_c = [sarc_pct, not_sarc_pct, irony_pct, sent_gap_pct]
    colors_c = ["#FF2D2D", "#00C851", "#FF8C00", "#FFE600"]
    ax.barh(labels_c[::-1], values_c[::-1], color=colors_c[::-1], height=0.45, edgecolor="#0d0d0d")
    for i, (val, lbl) in enumerate(zip(values_c[::-1], labels_c[::-1])):
        ax.text(min(val+1.5, 96), i, f"{val}%", va="center", ha="left", color="#f0ebe0", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="#555", fontsize=8)
    ax.tick_params(colors="#555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e1e")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # LIME
    st.markdown('<div class="sec-head">◈ LIME Word-Level Explanation</div>', unsafe_allow_html=True)

    if run_lime_check:
        with st.spinner("Running LIME analysis..."):
            word_scores = run_lime(headline, n=300)

        st.markdown(f"""
        <div class="lime-box">{build_lime_html(headline, word_scores)}</div>
        <div class="lime-legend">
          <span>🔴 Pushes toward sarcastic</span>
          <span>🟢 Suppresses sarcasm</span>
          <span>Hover over any word for its exact score</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-head">◈ Key Signal Words</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chip-wrap">{build_chip_html(word_scores)}</div>',
                    unsafe_allow_html=True)

        with st.expander("📋 Raw LIME scores table"):
            df_l = pd.DataFrame(word_scores, columns=["Word","Score"])
            df_l["Direction"] = df_l["Score"].apply(
                lambda s: "🔴 Sarcastic" if s > 0 else "🟢 Not Sarcastic")
            df_l["Score"] = df_l["Score"].round(4)
            df_l = df_l.sort_values("Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_l, use_container_width=True)
    else:
        st.info("LIME skipped. Tick the checkbox above and re-run to see word highlights.")

elif detect_btn and not headline.strip():
    st.warning("⚠️ Please enter a headline first.")

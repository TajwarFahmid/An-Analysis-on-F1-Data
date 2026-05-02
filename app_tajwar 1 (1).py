"""
F1 Points Finish Predictor
Individual Deployment — Tajwar Fahmid
DATA 4382 Capstone II

Run:
    pip install streamlit scikit-learn pandas numpy plotly
    streamlit run app_tajwar.py

Folder structure:
    app_tajwar.py
    data/
        races.csv, results.csv, qualifying.csv,
        constructor_standings.csv, driver_standings.csv, status.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="F1 Points Predictor · Tajwar Fahmid",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');
:root {
    --teal:#00B4D8; --navy:#0A1628; --green:#22C55E;
    --red:#E8002D;  --card:#0f2237; --border:#1a3352;
    --text:#e8f0f8; --muted:#6a8aaa; --gold:#FFD700;
}
html,body,[class*="css"]           { font-family:'Inter',sans-serif; background:var(--navy)!important; color:var(--text); }
[data-testid="stAppViewContainer"]  { background:var(--navy); }
[data-testid="stHeader"]            { background:var(--navy); }
[data-testid="stSidebar"]           { background:#050f1e!important; }
[data-testid="stSidebar"] *         { color:#8aaccc!important; }
[data-testid="stSidebar"] h2        { color:var(--teal)!important; font-family:'Barlow Condensed',sans-serif; text-transform:uppercase; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif; text-transform:uppercase; letter-spacing:.03em; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px 24px; margin-bottom:8px; }
.badge-yes { background:#22C55E22; color:#22C55E; border:1.5px solid #22C55E; border-radius:999px; padding:6px 20px; font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:700; display:inline-block; }
.badge-no  { background:#E8002D22; color:#E8002D; border:1.5px solid #E8002D; border-radius:999px; padding:6px 20px; font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:700; display:inline-block; }
.stButton>button { background:var(--teal)!important; color:#0A1628!important; font-family:'Barlow Condensed',sans-serif!important; font-size:1.1rem!important; font-weight:800!important; border:none!important; border-radius:8px!important; padding:12px 0!important; width:100%!important; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
FEATURES = ["grid", "pace_z_score", "team_pts_rolling_avg", "prev_driver_rank",
            "reliability_risk", "circuit_specialty", "grid_pace_delta", "rolling_pace"]

if not os.path.exists(DATA_DIR) or not os.path.exists(f"{DATA_DIR}/results.csv"):
    st.error("📁 Place Kaggle F1 CSV files in a `data/` folder next to this app.")
    st.stop()


@st.cache_data(show_spinner="Loading F1 dataset (2010–2023)…")
def load_data():
    races   = pd.read_csv(f"{DATA_DIR}/races.csv")[["raceId","year","round","circuitId","name"]]
    results = pd.read_csv(f"{DATA_DIR}/results.csv")[
                  ["raceId","driverId","constructorId","grid","positionOrder","statusId","points"]]
    qual    = pd.read_csv(f"{DATA_DIR}/qualifying.csv")[["raceId","driverId","position"]]
    drv_st  = pd.read_csv(f"{DATA_DIR}/driver_standings.csv")[["raceId","driverId","position","points"]]
    con_st  = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv")[["raceId","constructorId","position","points"]]
    status  = pd.read_csv(f"{DATA_DIR}/status.csv")[["statusId","status"]]

    modern_ids = set(races[races["year"] >= 2010]["raceId"])
    races_m    = races[races["year"] >= 2010][["raceId","year","round","circuitId","name"]].copy()
    races_m.rename(columns={"name":"race_name"}, inplace=True)

    res = results[results["raceId"].isin(modern_ids)].copy()
    res = res.merge(status, on="statusId", how="left")
    res["dnf"] = res["status"].str.contains(
        "Accident|Collision|Engine|Gearbox|Hydraulics|Mechanical|Retired|"
        "Suspension|Electrical|Brakes|Wheel|Oil|Water|Puncture|Fire|Clutch|Overheating",
        case=False, na=False).astype(int)
    res = res[["raceId","driverId","constructorId","grid","positionOrder","points","dnf"]].copy()

    qual_m = qual[qual["raceId"].isin(modern_ids)].copy()
    qual_m.columns = ["raceId","driverId","quali_pos"]

    drv_m = drv_st[drv_st["raceId"].isin(modern_ids)].copy()
    drv_m.columns = ["raceId","driverId","drv_rank","drv_pts"]

    con_m = con_st[con_st["raceId"].isin(modern_ids)].copy()
    con_m.columns = ["raceId","constructorId","con_rank","con_pts"]

    df = res.merge(races_m, on="raceId", how="left")
    df = df.merge(qual_m,  on=["raceId","driverId"],       how="left")
    df = df.merge(drv_m,   on=["raceId","driverId"],       how="left")
    df = df.merge(con_m,   on=["raceId","constructorId"],  how="left")

    df["grid"] = df["grid"].replace(0, np.nan).fillna(df["quali_pos"])
    df = df.dropna(subset=["grid","positionOrder"])
    df["grid"]          = df["grid"].astype(int)
    df["positionOrder"] = df["positionOrder"].astype(int)
    df = df.sort_values(["year","round","driverId"]).reset_index(drop=True)

    g = df.groupby("raceId")["grid"].agg(gm="mean", gs="std").reset_index()
    df = df.merge(g, on="raceId", how="left")
    df["pace_z_score"]  = -(df["grid"] - df["gm"]) / (df["gs"] + 1e-5)

    df["rolling_pace"] = -(
        df.groupby("driverId")["positionOrder"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
          .fillna(10))

    df["team_pts_rolling_avg"] = (
        df.groupby("constructorId")["con_pts"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
          .fillna(0))

    df["prev_driver_rank"] = df["drv_rank"].fillna(10)

    df["reliability_risk"] = (
        df.groupby("constructorId")["dnf"]
          .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
          .fillna(0.1))

    circ_avg = (
        df.groupby(["driverId","circuitId"])["positionOrder"]
          .transform(lambda x: x.shift(1).expanding().mean()))
    df["circuit_specialty"] = -(circ_avg.fillna(df["positionOrder"].mean()))

    df["grid_pace_delta"] = df["pace_z_score"] - (df["grid"] - 10) * 0.05
    df["scores_points"]   = (df["positionOrder"] <= 10).astype(int)

    keep = ["year","raceId","driverId","constructorId","race_name",
            "positionOrder","circuitId","scores_points"] + FEATURES
    return df[keep].dropna().reset_index(drop=True)


@st.cache_resource(show_spinner="Training Points Finish model…")
def train_model(_df):
    X     = _df[FEATURES].values
    y     = _df["scores_points"].values
    years = _df["year"].values

    tr = years <= 2022
    te = years == 2023
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    sw    = compute_sample_weight("balanced", y_tr)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_tr_s, y_tr, sample_weight=sw)

    pred_te = model.predict(X_te_s)
    prob_te = model.predict_proba(X_te_s)[:, 1]
    rep     = classification_report(y_te, pred_te,
                                    target_names=["P11+ (No Points)", "Top 10 (Points)"],
                                    output_dict=True)
    auc_sc  = roc_auc_score(y_te, prob_te)
    cm      = confusion_matrix(y_te, pred_te)

    tss     = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for ti, vi in tss.split(X_tr_s):
        m  = GradientBoostingClassifier(n_estimators=100, random_state=42)
        yt = y_tr[ti];  yv = y_tr[vi]
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            continue
        m.fit(X_tr_s[ti], yt)
        cv_aucs.append(roc_auc_score(yv, m.predict_proba(X_tr_s[vi])[:, 1]))

    # Points rate by grid — plain Python, no groupby/apply
    grid_col  = _df["grid"].values.astype(int)
    pts_col   = _df["scores_points"].values
    grid_vals = sorted(set(g for g in grid_col if 1 <= g <= 20))
    pts_rates = [float(pts_col[grid_col == g].mean() * 100) for g in grid_vals]
    pts_by_grid = pd.DataFrame({"grid": grid_vals, "Points Rate %": pts_rates})

    fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}
                      ).sort_values("importance", ascending=False).reset_index(drop=True)

    # Reliability sensitivity
    means     = _df[FEATURES].mean().values.copy()
    rel_idx   = FEATURES.index("reliability_risk")
    rel_vals  = np.linspace(0.0, 0.8, 30)
    rel_probs = []
    for rv in rel_vals:
        inp            = means.copy()
        inp[rel_idx]   = rv
        inp_s          = scaler.transform(inp.reshape(1, -1))
        rel_probs.append(float(model.predict_proba(inp_s)[0, 1] * 100))

    return dict(model=model, scaler=scaler, rep=rep, auc=auc_sc,
                cm=cm, cv_aucs=cv_aucs, pts_by_grid=pts_by_grid,
                fi=fi, rel_vals=rel_vals, rel_probs=rel_probs,
                feat_means=_df[FEATURES].mean(),
                feat_min=_df[FEATURES].min(),
                feat_q95=_df[FEATURES].quantile(0.95))


df     = load_data()
bundle = train_model(df)
model  = bundle["model"]
scaler = bundle["scaler"]

fm  = bundle["feat_means"]
fmn = bundle["feat_min"]
fq  = bundle["feat_q95"]

with st.sidebar:
    st.markdown("## 🎯 Points Predictor")
    st.markdown("Tajwar Fahmid · DATA 4382")
    st.markdown("---")
    st.markdown("**Will this driver score points (Top 10)?**")
    st.markdown("---")
    grid_pos  = st.slider("Grid / Qualifying Position", 1, 20, 8)
    pace_z    = st.slider("Pace Z-Score vs Field", -3.0, 3.0, 0.2, 0.05)
    team_pts  = st.slider("Constructor Rolling Pts Avg",
                          0.0, float(max(fq["team_pts_rolling_avg"], 1.0)),
                          float(fm["team_pts_rolling_avg"]))
    prev_rank = st.slider("Driver Championship Rank", 1, 20, 7)
    rel_risk  = st.slider("Reliability Risk", 0.0, 1.0, 0.1, 0.01)
    circ_spec = st.slider("Circuit Specialty Score",
                          float(fmn["circuit_specialty"]), 0.0,
                          float(fm["circuit_specialty"]), 0.1)
    gpd       = st.slider("Grid-Pace Delta", -3.0, 3.0,
                          round(float(pace_z - (grid_pos - 10) * 0.05), 2), 0.05)
    roll_pace = st.slider("Rolling Pace (last 3 races)",
                          float(fmn["rolling_pace"]), 0.0,
                          float(fm["rolling_pace"]), 0.1)
    st.markdown("---")
    st.button("🎯 Predict Points Finish")

st.markdown("""
<div style="background:linear-gradient(135deg,#0A1628 50%,#00131f);
            border-top:3px solid #00B4D8;border-radius:12px;
            padding:26px 32px;margin-bottom:18px;">
  <div style="font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
              color:#00B4D8;margin-bottom:6px;">
    DATA 4382 Capstone II · Individual Deployment · Tajwar Fahmid
  </div>
  <h1 style="color:#e8f0f8;margin:0 0 4px;font-size:1.9rem;">
    🎯 F1 Points Finish Predictor
  </h1>
  <div style="color:#6a8aaa;font-size:.88rem;">
    Model A — Points Filter · Will This Driver Score Points or Finish P11+?
  </div>
</div>
""", unsafe_allow_html=True)

rep = bundle["rep"]

def kpi(col, val, label, color):
    col.markdown(f"""
    <div class="card" style="border-left:4px solid {color};text-align:center;padding:14px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.9rem;font-weight:800;color:{color};">{val}</div>
      <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:#6a8aaa;margin-top:3px;">{label}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
kpi(c1, f"{bundle['auc']:.3f}", "Test AUC", "#00B4D8")
kpi(c2, f"{rep.get('accuracy',0)*100:.1f}%", "Overall Accuracy", "#22C55E")
kpi(c3, f"{rep.get('Top 10 (Points)',{}).get('f1-score',0)*100:.1f}%", "Points F1-Score", "#FFD700")
kpi(c4, f"{np.mean(bundle['cv_aucs']):.3f}" if bundle['cv_aucs'] else "N/A", "Mean CV AUC", "#6a8aaa")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🎯  Predict", "📊  Grid Analysis", "🔧  Reliability Impact", "📈  Performance"])

with tab1:
    inp_s = scaler.transform(
        np.array([grid_pos, pace_z, team_pts, prev_rank,
                  rel_risk, circ_spec, gpd, roll_pace]).reshape(1, -1))
    proba   = model.predict_proba(inp_s)[0]
    pred    = int(model.predict(inp_s)[0])
    pts_pct = float(proba[1] * 100)
    out_pct = float(proba[0] * 100)

    label = "Top 10 · Points Scored ✓" if pred == 1 else "P11+ · Outside Points"
    badge = "badge-yes"                if pred == 1 else "badge-no"
    conf  = pts_pct                    if pred == 1 else out_pct
    color = "#22C55E"                  if pred == 1 else "#E8002D"

    cr, cg = st.columns([1, 1.5])
    with cr:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {color};text-align:center;padding:32px 20px;">
          <div style="font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:#6a8aaa;margin-bottom:12px;">
            Prediction · Grid P{grid_pos}
          </div>
          <span class="{badge}">{label}</span>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:3.5rem;font-weight:800;color:{color};line-height:1;margin-top:18px;">
            {conf:.0f}%
          </div>
          <div style="font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;color:#6a8aaa;margin-top:4px;">Confidence</div>
          <div style="margin-top:18px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
              <span style="font-size:.85rem;color:#6a8aaa;">Points probability</span>
              <span style="font-size:.85rem;font-weight:600;color:#22C55E;">{pts_pct:.1f}%</span>
            </div>
            <div style="background:#1a3352;border-radius:4px;height:8px;">
              <div style="background:#22C55E;height:100%;width:{min(pts_pct,100):.0f}%;border-radius:4px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:8px;margin-bottom:4px;">
              <span style="font-size:.85rem;color:#6a8aaa;">P11+ probability</span>
              <span style="font-size:.85rem;font-weight:600;color:#E8002D;">{out_pct:.1f}%</span>
            </div>
            <div style="background:#1a3352;border-radius:4px;height:8px;">
              <div style="background:#E8002D;height:100%;width:{min(out_pct,100):.0f}%;border-radius:4px;"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with cg:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pts_pct,
            number={"suffix":"%","font":{"size":28,"color":"#e8f0f8"},"valueformat":".1f"},
            gauge={"axis":{"range":[0,100],"tickcolor":"#1a3352"},
                   "bar":{"color":"#22C55E" if pred==1 else "#E8002D","thickness":0.3},
                   "bgcolor":"#0A1628","borderwidth":0,
                   "steps":[{"range":[0,40],"color":"#0f1e30"},
                             {"range":[40,60],"color":"#0a2030"},
                             {"range":[60,100],"color":"#0a2510"}],
                   "threshold":{"line":{"color":"#FFD700","width":2},"thickness":0.75,"value":50}},
            title={"text":"Points Probability","font":{"size":12,"color":"#6a8aaa"}},
        ))
        fig_g.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#0f2237")
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"""
        <div class="card" style="border-left:3px solid #00B4D8;padding:12px 16px;">
          <div style="font-size:.78rem;font-weight:600;color:#00B4D8;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Key Inputs</div>
          <div style="font-size:.85rem;color:#8aaccc;line-height:1.6;">
            Grid P{grid_pos} · Pace Z: {pace_z:+.2f}<br>
            Reliability: {rel_risk:.2f} · Champ Rank: #{prev_rank}<br>
            Constructor Pts Avg: {team_pts:.0f}
          </div>
        </div>""", unsafe_allow_html=True)

    if pred == 1 and pts_pct < 60:
        st.warning("Borderline — small race-day changes could push this driver outside the points.")
    if rel_risk > 0.3:
        st.error("High reliability risk flagged — significant DNF history for this constructor.")
    st.markdown("""
    <div style="background:#0f2237;border-left:3px solid #6a8aaa;border-radius:0 8px 8px 0;padding:10px 16px;margin-top:12px;">
      <div style="font-size:.83rem;color:#6a8aaa;">
        <strong style="color:#8aaccc;">Limitation:</strong> Pre-race prediction only.
        Safety cars, crashes, and mechanical failures during the race are not predictable from qualifying data.
      </div>
    </div>""", unsafe_allow_html=True)

with tab2:
    pgrid = bundle["pts_by_grid"]
    fig_grid = go.Figure()
    fig_grid.add_trace(go.Bar(
        x=pgrid["grid"].tolist(), y=pgrid["Points Rate %"].tolist(),
        marker=dict(color=pgrid["Points Rate %"].tolist(),
                    colorscale=[[0,"#E8002D"],[0.5,"#FFD700"],[1,"#22C55E"]],
                    showscale=False),
        text=[f"{v:.0f}%" for v in pgrid["Points Rate %"].tolist()],
        textposition="outside", textfont=dict(color="#8aaccc", size=9),
    ))
    fig_grid.update_layout(
        title="Actual Top-10 Finish Rate by Grid Position — F1 2010–2023",
        xaxis=dict(title="Grid Position", dtick=1, color="#3a5a7a"),
        yaxis=dict(title="Points Finish Rate (%)", color="#3a5a7a",
                   gridcolor="#1a3352", range=[0, 115]),
        height=360, paper_bgcolor="#0f2237", plot_bgcolor="#0f2237",
        font=dict(family="Inter", size=11, color="#6a8aaa"),
        margin=dict(l=20,r=20,t=40,b=20),
    )
    st.plotly_chart(fig_grid, use_container_width=True)

    fi = bundle["fi"]
    fig_fi = go.Figure(go.Bar(
        x=fi["importance"].values[::-1], y=fi["feature"].values[::-1],
        orientation="h",
        marker=dict(color=fi["importance"].values[::-1],
                    colorscale=[[0,"#1a3352"],[0.5,"#006080"],[1,"#00B4D8"]],
                    showscale=False),
        text=[f"{v:.3f}" for v in fi["importance"].values[::-1]],
        textposition="outside", textfont=dict(color="#00B4D8", size=10),
    ))
    fig_fi.update_layout(
        title="Feature Importance — Model A (Points Filter)",
        xaxis=dict(title="Importance Score", color="#3a5a7a", gridcolor="#1a3352"),
        yaxis=dict(color="#3a5a7a"),
        height=340, paper_bgcolor="#0f2237", plot_bgcolor="#0f2237",
        font=dict(family="Inter", size=11, color="#6a8aaa"),
        margin=dict(l=20,r=20,t=40,b=20),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with tab3:
    st.write("Reliability risk has a near-linear negative effect on points probability — "
             "a team with a 30% DNF rate loses ~20–25 percentage points regardless of grid position.")
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=(bundle["rel_vals"] * 100).tolist(), y=bundle["rel_probs"],
        mode="lines", line=dict(color="#00B4D8", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
    ))
    fig_rel.add_vline(x=rel_risk*100, line_dash="dot", line_color="#FFD700",
                      annotation_text=f"Your input: {rel_risk:.0%}",
                      annotation_font_color="#FFD700", annotation_position="top left")
    fig_rel.update_layout(
        title="Points Probability vs Reliability Risk (all other inputs at dataset mean)",
        xaxis=dict(title="Reliability Risk (%)", color="#3a5a7a", gridcolor="#1a3352"),
        yaxis=dict(title="Points Probability (%)", color="#3a5a7a",
                   gridcolor="#1a3352", range=[0, 100]),
        height=340, paper_bgcolor="#0f2237", plot_bgcolor="#0f2237",
        font=dict(family="Inter", size=11, color="#6a8aaa"),
        margin=dict(l=20,r=20,t=40,b=20),
    )
    st.plotly_chart(fig_rel, use_container_width=True)

with tab4:
    col_cv, col_cm = st.columns(2)
    with col_cv:
        cv = bundle["cv_aucs"]
        if cv:
            mn = float(np.mean(cv))
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv))],
                y=[float(v) for v in cv],
                marker_color=["#E8002D" if v < mn else "#00B4D8" for v in cv],
                text=[f"{v:.3f}" for v in cv], textposition="outside",
                textfont=dict(color="#8aaccc"),
            ))
            fig_cv.add_hline(y=mn, line_dash="dash", line_color="#FFD700",
                             annotation_text=f"Mean {mn:.3f}",
                             annotation_font_color="#FFD700", annotation_position="top right")
            fig_cv.update_layout(
                title="5-Fold Time-Series CV AUC",
                yaxis=dict(range=[max(0.5,min(cv)-0.05),1.0],
                           title="AUC", color="#3a5a7a", gridcolor="#1a3352"),
                xaxis=dict(color="#3a5a7a"),
                height=300, paper_bgcolor="#0f2237", plot_bgcolor="#0f2237",
                font=dict(family="Inter", size=11, color="#6a8aaa"),
                margin=dict(l=20,r=20,t=40,b=20),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    with col_cm:
        cm = bundle["cm"]
        fig_cm = px.imshow(cm, text_auto=True,
                           x=["P11+","Top 10"], y=["P11+","Top 10"],
                           color_continuous_scale=[[0,"#0a1628"],[0.5,"#005580"],[1,"#00B4D8"]],
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix — 2023 Test Season")
        fig_cm.update_layout(height=300, paper_bgcolor="#0f2237",
                             font=dict(family="Inter", size=11, color="#6a8aaa"),
                             margin=dict(l=20,r=20,t=40,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    rows = []
    for cls in ["P11+ (No Points)","Top 10 (Points)"]:
        if cls in rep:
            r = rep[cls]
            rows.append({"Class": cls, "Precision": f"{r['precision']:.3f}",
                         "Recall": f"{r['recall']:.3f}", "F1-Score": f"{r['f1-score']:.3f}",
                         "Support": int(r["support"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""<div style="text-align:center;color:#1a3352;font-size:.78rem;padding:8px 0;">
DATA 4382 – Capstone II · Tajwar Fahmid · Model A: Points Filter · Kaggle F1 (Rohan Rao, 2010–2023)
</div>""", unsafe_allow_html=True)

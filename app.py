from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from src.data_loader import load_data
from src.preprocessing import clean_data, dataset_summary
from src.feature_engineering import build_player_features, add_recency_scores
from src.player_profiles import build_profile_scores
from src.hybrid_recommender import AuctionIQRecommender
from src.explainability import explain_recommendation
from src.ranking import score_breakdown


st.set_page_config(
    page_title="AuctionIQ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Global UI theme
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --iq-bg: #07111f;
        --iq-panel: #0d1b2d;
        --iq-panel-2: #11253b;
        --iq-text: #f5f7fb;
        --iq-muted: #9db0c5;
        --iq-accent: #ffcc66;
        --iq-accent-2: #78c7ff;
        --iq-border: rgba(255,255,255,0.09);
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(120,199,255,.10), transparent 25%),
            radial-gradient(circle at 100% 10%, rgba(255,204,102,.08), transparent 22%),
            var(--iq-bg);
        color: var(--iq-text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091525 0%, #07111f 100%);
        border-right: 1px solid var(--iq-border);
    }

    [data-testid="stSidebarContent"] { padding-top: 1rem; }

    .iq-brand {
        padding: 16px 8px 20px 8px;
        border-bottom: 1px solid var(--iq-border);
        margin-bottom: 16px;
    }
    .iq-brand h2 { margin: 0; font-size: 1.35rem; }
    .iq-brand p { margin: 4px 0 0; color: var(--iq-muted); font-size: .82rem; }

    .iq-hero {
        padding: 26px 28px;
        border: 1px solid var(--iq-border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(17,37,59,.96), rgba(9,21,37,.95));
        box-shadow: 0 18px 60px rgba(0,0,0,.20);
        margin-bottom: 20px;
    }
    .iq-hero .eyebrow {
        color: var(--iq-accent);
        text-transform: uppercase;
        letter-spacing: .15em;
        font-weight: 700;
        font-size: .75rem;
    }
    .iq-hero h1 { margin: 7px 0 8px; font-size: 2.35rem; }
    .iq-hero p { margin: 0; color: var(--iq-muted); font-size: 1rem; max-width: 900px; }

    .iq-card {
        border: 1px solid var(--iq-border);
        border-radius: 18px;
        background: rgba(13,27,45,.82);
        padding: 18px;
        height: 100%;
    }
    .iq-card .label { color: var(--iq-muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .iq-card .value { font-size: 1.75rem; font-weight: 800; margin-top: 5px; }
    .iq-card .hint { color: var(--iq-muted); font-size: .78rem; margin-top: 3px; }

    .rank-card {
        border: 1px solid var(--iq-border);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(17,37,59,.95), rgba(10,24,40,.92));
    }
    .rank-num { color: var(--iq-accent); font-weight: 800; font-size: 1.15rem; }
    .rank-name { font-weight: 800; font-size: 1rem; }
    .rank-role { color: var(--iq-muted); font-size: .78rem; }
    .rank-score { font-size: 1.35rem; font-weight: 800; text-align: right; }
    .rank-confidence { color: var(--iq-accent-2); font-size: .75rem; text-align: right; }

    .section-title { font-size: 1.2rem; font-weight: 800; margin: 12px 0 10px; }
    .muted { color: var(--iq-muted); }

    div[data-testid="stMetric"] {
        background: rgba(13,27,45,.82);
        border: 1px solid var(--iq-border);
        border-radius: 16px;
        padding: 14px;
    }

    .stButton > button[kind="primary"] {
        border-radius: 12px;
        font-weight: 800;
        min-height: 45px;
    }

    /* Compact hamburger navigation feel */
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading IPL data and building player profiles…")
def prepare():
    deliveries, matches = load_data(ROOT / "data")
    deliveries, matches = clean_data(deliveries, matches)
    features = build_player_features(deliveries, matches)
    features = add_recency_scores(features, deliveries, matches, "Last 2 seasons")
    profiles = build_profile_scores(features)
    numeric_cols = [c for c in profiles.columns if profiles[c].dtype != "object"]
    profiles["feature_completeness"] = 100 * profiles[numeric_cols].notna().mean(axis=1)
    recommender = AuctionIQRecommender(profiles)
    return deliveries, matches, profiles, recommender


deliveries, matches, profiles, recommender = prepare()
summary = dataset_summary(deliveries, matches)


# -----------------------------
# Sidebar / hamburger-style nav
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="iq-brand">
            <h2>🏏 AuctionIQ</h2>
            <p>IPL player recommendation intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ☰ Menu")
    page = st.radio(
        "",
        [
            "⌂  Dashboard",
            "🎯  Auction Recommender",
            "🔎  Player Explorer",
            "🧩  Similar Players",
            "⚔️  Compare Players",
            "🏟️  Squad Analyzer",
            "📊  Model Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("DATASET")
    st.write(f"{summary['matches']:,} matches")
    st.write(f"{summary['deliveries']:,} deliveries")
    st.write(f"{summary['players']:,} players")
    st.caption("Recommendation engine: hybrid + constraints + diversity")


# -----------------------------
# Helpers
# -----------------------------
def hero(title, subtitle, eyebrow="AUCTIONIQ"):
    st.markdown(
        f"""
        <div class="iq-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="iq-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def player_rank_cards(out):
    for _, row in out.iterrows():
        st.markdown(
            f"""
            <div class="rank-card">
              <div style="display:grid;grid-template-columns:60px 1fr 120px;gap:12px;align-items:center;">
                <div class="rank-num">#{int(row['rank'])}</div>
                <div>
                    <div class="rank-name">{row['player']}</div>
                    <div class="rank-role">{row['role']}</div>
                </div>
                <div>
                    <div class="rank-score">{row['final_score']:.1f}</div>
                    <div class="rank-confidence">{row['confidence']} confidence</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Dashboard
# -----------------------------
if page.startswith("⌂"):
    hero(
        "Build a better auction shortlist.",
        "AuctionIQ converts historical IPL performance into personalized player recommendations — built around team fit, not just raw statistics.",
    )

    metric_cards([
        ("Matches", f"{summary['matches']:,}", "historical IPL matches"),
        ("Deliveries", f"{summary['deliveries']:,}", "ball-by-ball records"),
        ("Seasons", f"{summary['seasons']:,}", "covered in supplied data"),
        ("Players", f"{summary['players']:,}", "player profiles generated"),
    ])

    st.markdown("<div class='section-title'>Recommendation workspace</div>", unsafe_allow_html=True)
    a, b = st.columns([1.55, 1])
    with a:
        st.markdown(
            """
            <div class="iq-card">
              <div class="label">How the engine thinks</div>
              <h3 style="margin:.3rem 0 .7rem">Requirement → Profile → Ranking</h3>
              <p class="muted">Hard constraints narrow the candidate pool. Soft preferences score team fit. Content similarity, recent form, consistency and diversity refine the final Top-K list.</p>
              <p style="margin-bottom:0"><b>Core idea:</b> a high-performing player is not automatically the best recommendation for every team.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="iq-card">
              <div class="label">Start here</div>
              <h3 style="margin:.3rem 0 .7rem">🎯 Auction Recommender</h3>
              <p class="muted">Set role, batting/bowling priorities, form, consistency and risk. Generate a personalized shortlist and inspect the explanation for every player.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Explore the data</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        top_runs = profiles.nlargest(10, "runs")[["player", "runs"]]
        fig = px.bar(top_runs.sort_values("runs"), x="runs", y="player", orientation="h", title="Top run scorers")
        fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=45,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_wkts = profiles.nlargest(10, "wickets")[["player", "wickets"]]
        fig = px.bar(top_wkts.sort_values("wickets"), x="wickets", y="player", orientation="h", title="Top wicket takers")
        fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=45,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Auction Recommender
# -----------------------------
elif page.startswith("🎯"):
    hero(
        "Auction Recommender",
        "Define what your team needs. AuctionIQ ranks players by fit, not by career totals alone.",
        "PERSONALIZED RECOMMENDATION",
    )

    with st.container(border=True):
        st.markdown("### Team profile")
        c1, c2 = st.columns(2)
        with c1:
            required_role = st.selectbox(
                "Required role",
                ["Any", "Batter", "Top/Middle-Order Batter", "Finisher", "All-Rounder", "Bowling Specialist", "Bowling Part-Time / All-Rounder"],
            )
            batting = st.slider("Batting importance", 0, 100, 60)
            bowling = st.slider("Bowling importance", 0, 100, 60)
            recent = st.slider("Recent form importance", 0, 100, 70)
        with c2:
            consistency = st.slider("Consistency importance", 0, 100, 60)
            risk = st.selectbox("Risk preference", ["Low", "Medium", "High"], index=1)
            topk = st.selectbox("Top-K shortlist", [5, 10, 15, 20], index=0)
            budget = st.number_input("Planning budget (₹ Cr)", min_value=0.0, value=10.0, step=0.5)
            st.caption("Budget is a user planning input only; the supplied datasets contain no historical auction-price field.")

        if st.button("Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner("Personalizing shortlist…"):
                st.session_state["recommendations"] = recommender.recommend(
                    required_role=required_role,
                    batting_importance=max(batting / 100, 0.01),
                    bowling_importance=max(bowling / 100, 0.01),
                    recent_importance=max(recent / 100, 0.01),
                    consistency_importance=max(consistency / 100, 0.01),
                    risk=risk,
                    top_k=topk,
                )
                st.session_state["requirement_snapshot"] = {
                    "role": required_role,
                    "batting": batting,
                    "bowling": bowling,
                    "recent": recent,
                    "consistency": consistency,
                    "risk": risk,
                    "budget": budget,
                }

    out = st.session_state.get("recommendations")
    if out is not None and not out.empty:
        st.markdown("<div class='section-title'>Your shortlist</div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Best Fit", out.iloc[0]["player"])
        k2.metric("Recommendation Score", f"{out.iloc[0]['final_score']:.1f}")
        k3.metric("Confidence", out.iloc[0]["confidence"])

        left, right = st.columns([.95, 1.6])
        with left:
            player_rank_cards(out.head(topk))
        with right:
            show = out[["rank", "player", "final_score", "confidence"]].rename(columns={"rank":"Rank", "player":"Player", "final_score":"Score", "confidence":"Confidence"})
            fig = px.bar(show.sort_values("Score"), x="Score", y="Player", orientation="h", text_auto=".1f", title="Personalized ranking")
            fig.update_layout(template="plotly_dark", height=max(420, min(720, 75 * len(show))), margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Recommendation explanation</div>", unsafe_allow_html=True)
        selected = st.selectbox("Inspect player", out["player"].tolist(), label_visibility="collapsed")
        row = out[out["player"].eq(selected)].iloc[0]
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Final score", f"{row['final_score']:.1f}")
        x2.metric("Role", row["role"])
        x3.metric("Confidence", row["confidence"])
        x4.metric("Form", f"{row['recent_form_score']:.1f}")

        reason_cols = st.columns(2)
        with reason_cols[0]:
            st.markdown("#### Why recommended?")
            for reason in explain_recommendation(row):
                st.write("✓", reason)
        with reason_cols[1]:
            st.markdown("#### Score breakdown")
            bd = pd.DataFrame(score_breakdown(row).items(), columns=["Component", "Score"])
            st.dataframe(bd, use_container_width=True, hide_index=True)


# -----------------------------
# Player Explorer
# -----------------------------
elif page.startswith("🔎"):
    hero("Player Explorer", "Search, filter and inspect the generated player profiles.", "PLAYER INTELLIGENCE")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: q = st.text_input("Search player name")
        with c2: role = st.selectbox("Role", ["All"] + sorted(profiles["role"].unique().tolist()))
        with c3: min_apps = st.slider("Minimum appearances", 0, int(profiles["appearances"].max()), 0)

    temp = profiles.copy()
    if q: temp = temp[temp["player"].str.contains(q, case=False, na=False)]
    if role != "All": temp = temp[temp["role"].eq(role)]
    temp = temp[temp["appearances"] >= min_apps]

    metric_cards([
        ("Players found", f"{len(temp):,}", "matching current filters"),
        ("Avg runs", f"{temp['runs'].mean():.1f}" if len(temp) else "—", "within filtered set"),
        ("Avg wickets", f"{temp['wickets'].mean():.1f}" if len(temp) else "—", "within filtered set"),
    ])
    cols = ["player","role","appearances","runs","strike_rate","batting_average","wickets","economy","recent_form_score","consistency_score"]
    st.dataframe(temp[cols].sort_values("appearances", ascending=False), use_container_width=True, hide_index=True)


# -----------------------------
# Similar Players
# -----------------------------
elif page.startswith("🧩"):
    hero("Similar Players", "Find players with comparable statistical profiles using cosine similarity.", "CONTENT-BASED FILTERING")
    c1, c2 = st.columns([2,1])
    with c1: player = st.selectbox("Player", profiles["player"].sort_values().tolist())
    with c2: k = st.selectbox("Top-K", [5,10,15], index=0)
    sim = recommender.similar(player, k)
    if not sim.empty:
        a,b = st.columns([1.25,1])
        with a: st.dataframe(sim, use_container_width=True, hide_index=True)
        with b:
            fig = px.bar(sim.sort_values("similarity"), x="similarity", y="player", orientation="h", text_auto=".2f", title=f"Similarity to {player}")
            fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=50,b=0), height=380)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Similarity is computed from the recommender's normalized content feature space; it is not a claim that two players have identical real-world roles.")


# -----------------------------
# Compare Players
# -----------------------------
elif page.startswith("⚔️"):
    hero("Compare Players", "Compare two to four candidates side-by-side using the same signals used by the recommender.", "DECISION SUPPORT")
    selected = st.multiselect("Select 2–4 players", profiles["player"].sort_values().tolist(), max_selections=4)
    if len(selected) >= 2:
        cols = ["runs","strike_rate","batting_average","wickets","economy","recent_form_score","consistency_score","role"]
        cmp = profiles[profiles["player"].isin(selected)][["player"] + cols].set_index("player").T
        st.dataframe(cmp, use_container_width=True)
        melted = profiles[profiles["player"].isin(selected)][["player","batting_strength","bowling_strength","recent_form_score","consistency_score"]].melt(
            id_vars="player", var_name="metric", value_name="score"
        )
        fig = px.bar(melted, x="metric", y="score", color="player", barmode="group", title="Profile signal comparison")
        fig.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Choose at least two players to start the comparison.")


# -----------------------------
# Squad Analyzer
# -----------------------------
elif page.startswith("🏟️"):
    hero("Squad Analyzer", "Inspect historical franchise participation and surface potential role gaps for recommendation workflows.", "TEAM CONTEXT")
    st.info("This is based on the supplied historical match/delivery data. It is not a current 2026 squad roster.")
    teams = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))
    team = st.selectbox("Franchise", teams)
    team_del = deliveries[(deliveries["team_batting"].eq(team)) | (deliveries["team_bowling"].eq(team))]
    player_counts = pd.concat([
        team_del[team_del["team_batting"].eq(team)].groupby("batter")["match_id"].nunique().rename("bat_matches"),
        team_del[team_del["team_bowling"].eq(team)].groupby("bowler")["match_id"].nunique().rename("bowl_matches"),
    ], axis=1).fillna(0)
    player_counts["appearances"] = player_counts.max(axis=1)
    top = player_counts.sort_values("appearances", ascending=False).head(12).reset_index().rename(columns={"index":"player"})
    st.dataframe(top, use_container_width=True, hide_index=True)
    role_mix = profiles[profiles["player"].isin(top.iloc[:,0])]["role"].value_counts().reset_index()
    role_mix.columns = ["role", "players"]
    fig = px.bar(role_mix, x="role", y="players", title=f"Historical role mix — {team}")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Model Insights
# -----------------------------
elif page.startswith("📊"):
    hero("Model Insights", "Understand what powers the recommendations and how the recommender is assembled.", "RECOMMENDER TRANSPARENCY")
    metric_cards([
        ("Content features", f"{len(recommender.content_features):,}", "used in content representation"),
        ("Player profiles", f"{len(profiles):,}", "candidate records"),
        ("Similarity metric", "Cosine", "normalized content space"),
        ("Ranking", "Hybrid", "fit + form + consistency + similarity"),
    ])
    left, right = st.columns(2)
    with left:
        st.markdown("### Hybrid weighting")
        st.dataframe(pd.DataFrame({"Signal":["Content-Based","Team Requirement","Recent Form","Consistency","Similarity","Preference"],"Default Weight":[.30,.30,.15,.10,.10,.05]}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Content features")
        st.code("\n".join(recommender.content_features), language="text")

    feat = profiles[["batting_strength","bowling_strength","recent_form_score","consistency_score"]].mean().reset_index()
    feat.columns = ["Feature","Mean score"]
    fig = px.bar(feat, x="Feature", y="Mean score", title="Average normalized player-profile strength")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Method notes")
    st.write("The dashboard deliberately separates recommendation from raw analytics. Historical team-player participation can support a collaborative extension, but auction price, nationality and official role are not fabricated when absent from the supplied files.")

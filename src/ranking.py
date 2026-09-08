from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DEFAULT_WEIGHTS

def normalize_100(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return 100*(s-lo)/(hi-lo)

def role_match(profile_role: str, required_role: str) -> float:
    if required_role in ("Any", None):
        return 100.0
    if profile_role == required_role:
        return 100.0
    # Broad compatibility is deliberately transparent rather than hard-coded to every cricket label.
    if "All-Rounder" in required_role and ("All-Rounder" in profile_role or "Part-Time" in profile_role):
        return 75.0
    if required_role == "Batter" and "Batter" in profile_role:
        return 80.0
    if required_role == "Bowler" and "Bowling" in profile_role:
        return 80.0
    return 20.0

def calculate_recommendations(
    profiles: pd.DataFrame,
    required_role="Any",
    batting_importance=0.5,
    bowling_importance=0.5,
    recent_importance=0.5,
    consistency_importance=0.5,
    risk="Medium",
    top_k=5,
    diversity_lambda=0.8,
    weights=None,
    similarity_seed=None,
):
    p = profiles.copy()

    p["role_match"] = p["role"].map(lambda x: role_match(x, required_role))
    p["batting_match"] = p["batting_strength"].clip(0,100)
    p["bowling_match"] = p["bowling_strength"].clip(0,100)
    # User preference vector: reweight batting/bowling and importance sliders.
    p["requirement_score"] = (
        batting_importance * p["batting_match"] +
        bowling_importance * p["bowling_match"] +
        0.2 * p["role_match"] +
        recent_importance * p["recent_form_score"] +
        consistency_importance * p["consistency_score"]
    ) / (batting_importance + bowling_importance + 0.2 + recent_importance + consistency_importance)
    # Conservative risk proxy: combines experience and consistency.
    target = {"Low":0.80,"Medium":0.60,"High":0.40}.get(risk,0.60)
    p["risk_fit"] = (1 - abs((p["experience_score"]/100*0.5 + p["consistency_score"]/100*0.5) - target))*100
    p["preference_score"] = (
        0.45*p["batting_match"]*(batting_importance) +
        0.45*p["bowling_match"]*(bowling_importance) +
        0.10*p["experience_score"]
    ).clip(0,100)

    p["similarity_score"] = 0.0 if similarity_seed is None else similarity_seed.reindex(p["player"]).fillna(0).values

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    # Weights sum to 1; normalize in case experiments use arbitrary totals.
    total = sum(w.values())
    w = {k:v/total for k,v in w.items()}

    p["content_score"] = 0.5*p["batting_strength"] + 0.5*p["bowling_strength"]
    p["constraint_compatibility"] = p["role_match"]
    p["final_score_raw"] = (
        w["content"]*p["content_score"] +
        w["requirement"]*p["requirement_score"] +
        w["recent_form"]*p["recent_form_score"] +
        w["consistency"]*p["consistency_score"] +
        w["similarity"]*p["similarity_score"] +
        w["preference"]*p["preference_score"]
    )
    p["final_score"] = p["final_score_raw"].clip(0,100)
    # Data coverage/confidence
    coverage = 0.45*p["experience_score"] + 0.35*p["consistency_score"] + 0.20*p["feature_completeness"].fillna(100)
    p["confidence"] = np.select([coverage>=75, coverage>=50], ["High","Medium"], default="Low")
    p = p.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected = []
    selected_indices = []
    for i, row in p.iterrows():
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(row.to_dict()); selected_indices.append(i); continue
        # MMR-style diversity using available scalar profile signature.
        candidate_vec = np.array([row.get(c,0) for c in ["batting_strength","bowling_strength","recent_form_score","consistency_score","powerplay_score","middle_score","death_score"]], dtype=float)
        rel = row["final_score"] / 100
        max_sim = 0
        for j in selected:
            vec = np.array([j.get(c,0) for c in ["batting_strength","bowling_strength","recent_form_score","consistency_score","powerplay_score","middle_score","death_score"]], dtype=float)
            denom = np.linalg.norm(candidate_vec)*np.linalg.norm(vec)
            sim = float(candidate_vec.dot(vec)/denom) if denom else 0
            max_sim = max(max_sim, sim)
        mmr = diversity_lambda*rel - (1-diversity_lambda)*max_sim
        if mmr >= -0.15 or len(p)-i <= (top_k-len(selected)):
            selected.append(row.to_dict()); selected_indices.append(i)
    out = pd.DataFrame(selected)
    out["rank"] = range(1, len(out)+1)
    return out

def score_breakdown(row):
    return {
        "Requirement Match": round(float(row.get("requirement_score",0)),1),
        "Content Match": round(float(row.get("content_score",0)),1),
        "Recent Form": round(float(row.get("recent_form_score",0)),1),
        "Consistency": round(float(row.get("consistency_score",0)),1),
        "Similarity": round(float(row.get("similarity_score",0)),1),
        "Preference Match": round(float(row.get("preference_score",0)),1),
        "Final Score": round(float(row.get("final_score",0)),1),
    }

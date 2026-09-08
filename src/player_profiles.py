from __future__ import annotations
import numpy as np
import pandas as pd

def minmax_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return 100 * (s - lo) / (hi - lo)

def build_profile_scores(features: pd.DataFrame) -> pd.DataFrame:
    p = features.copy()
    p["batting_strength"] = (
        0.35 * minmax_100(p["runs"]) +
        0.25 * minmax_100(p["strike_rate"]) +
        0.20 * minmax_100(p["batting_average"]) +
        0.20 * minmax_100(p["boundary_pct"])
    )
    # Lower economy is better.
    econ = p["economy"].replace(0, np.nan)
    econ_score = 100 * (econ.max(skipna=True) - econ) / max(econ.max(skipna=True) - econ.min(skipna=True), 1e-9)
    p["bowling_strength"] = (
        0.35 * minmax_100(p["wickets"]) +
        0.25 * minmax_100(p["dot_ball_pct"]) +
        0.20 * econ_score.fillna(0) +
        0.20 * minmax_100(p["wickets_per_match"])
    ).clip(0,100)
    p["experience_score"] = minmax_100(p["appearances"])
    p["consistency_score"] = (
        0.6 * (100 - minmax_100(p["runs_std"])) +
        0.4 * (100 - minmax_100(p["economy_std"]))
    ).clip(0,100)
    p["recent_form_score"] = p.get("recent_form_score", minmax_100(p["avg_recent_bat"] + p["avg_recent_wkts"])).clip(0,100)
    p["powerplay_score"] = (
        0.5 * minmax_100(p.get("powerplay_runs", pd.Series(0,index=p.index))) +
        0.5 * minmax_100(p.get("powerplay_wickets", pd.Series(0,index=p.index)))
    )
    p["middle_score"] = (
        0.5 * minmax_100(p.get("middle_runs", pd.Series(0,index=p.index))) +
        0.5 * minmax_100(p.get("middle_wickets", pd.Series(0,index=p.index)))
    )
    p["death_score"] = (
        0.33 * minmax_100(p.get("death_runs", pd.Series(0,index=p.index))) +
        0.34 * minmax_100(p.get("death_wickets", pd.Series(0,index=p.index))) +
        0.33 * (100 - minmax_100(p.get("death_economy", pd.Series(0,index=p.index))))
    ).clip(0,100)
    return p

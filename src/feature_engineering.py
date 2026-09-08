from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PHASE_BOUNDS, WICKET_TYPES_CREDITED_TO_BOWLER

def _safe_div(a, b):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.divide(a_arr, b_arr, out=np.zeros_like(a_arr, dtype=float), where=b_arr != 0)

def _base_player_ids(d):
    return pd.Index(sorted(set(d["batter"].dropna()) | set(d["bowler"].dropna()) | set(d["non_striker"].dropna())), name="player")

def build_player_features(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    m = matches.copy()

    match_dates = m[["id", "date", "season"]].rename(columns={"id": "match_id"})
    d = d.merge(match_dates, on="match_id", how="left", validate="many_to_one")
    d["phase"] = np.select(
        [d["over"].between(0, 5), d["over"].between(6, 14), d["over"].between(15, 19)],
        ["Powerplay", "Middle", "Death"],
        default="Other",
    )

    players = pd.DataFrame(index=_base_player_ids(d)).reset_index()

    # Batting
    bat = d[d["batter"].notna()].copy()
    bat["balls_faced_flag"] = (~bat["extras_type"].eq("wides")).astype(int)
    batting = bat.groupby("batter").agg(
        matches=("match_id", "nunique"),
        runs=("batsman_runs", "sum"),
        balls_faced=("balls_faced_flag", "sum"),
        fours=("batsman_runs", lambda s: int((s == 4).sum())),
        sixes=("batsman_runs", lambda s: int((s == 6).sum())),
    ).rename_axis("player").reset_index()
    # Innings are batter+match pairs where at least one delivery is faced.
    innings = bat.groupby("batter")["match_id"].nunique().rename("innings").reset_index()
    batting = batting.merge(innings, left_on="player", right_on="batter", how="left").drop(columns=["batter"])
    # Dismissal-based batting average is computed after merging dismissal counts.
    dismissals = d[d["player_dismissed"].notna()].groupby("player_dismissed").size().rename("dismissals")
    batting = batting.merge(dismissals, left_on="player", right_index=True, how="left")
    batting["dismissals"] = batting["dismissals"].fillna(0)
    batting["batting_average"] = _safe_div(batting["runs"], batting["dismissals"])
    batting["strike_rate"] = 100 * _safe_div(batting["runs"], batting["balls_faced"])
    batting["boundary_runs"] = batting["fours"] * 4 + batting["sixes"] * 6
    batting["boundary_pct"] = 100 * _safe_div(batting["boundary_runs"], batting["runs"])
    batting["runs_per_innings"] = _safe_div(batting["runs"], batting["innings"])
    batting["runs_per_match"] = _safe_div(batting["runs"], batting["matches"])

    # Bowling
    bowl = d[d["bowler"].notna()].copy()
    bowl["legal_ball"] = (~bowl["extras_type"].isin(["wides", "noballs"])).astype(int)
    bowl["bowler_runs"] = bowl["total_runs"] - bowl["extras_type"].isin(["byes", "legbyes", "penalty"]).astype(int) * bowl["total_runs"]
    bowl["bowler_runs"] = bowl["bowler_runs"].clip(lower=0)
    bowl["credited_wicket"] = (
        bowl["is_wicket"].eq(1) &
        bowl["dismissal_kind"].isin(WICKET_TYPES_CREDITED_TO_BOWLER)
    ).astype(int)
    bowling = bowl.groupby("bowler").agg(
        balls_bowled=("legal_ball", "sum"),
        runs_conceded=("bowler_runs", "sum"),
        wickets=("credited_wicket", "sum"),
        bowl_matches=("match_id", "nunique"),
        dot_balls=("total_runs", lambda s: int((s == 0).sum())),
    ).rename_axis("player").reset_index()
    bowling["overs"] = bowling["balls_bowled"] / 6
    bowling["economy"] = _safe_div(bowling["runs_conceded"] * 6, bowling["balls_bowled"])
    bowling["bowling_strike_rate"] = _safe_div(bowling["balls_bowled"], bowling["wickets"])
    bowling["dot_ball_pct"] = 100 * _safe_div(bowling["dot_balls"], bowling["balls_bowled"])
    bowling["wickets_per_match"] = _safe_div(bowling["wickets"], bowling["bowl_matches"])

    # Overall participation
    appearances = pd.concat([
        d[["match_id", "batter"]].rename(columns={"batter": "player"}),
        d[["match_id", "bowler"]].rename(columns={"bowler": "player"}),
        d[["match_id", "non_striker"]].rename(columns={"non_striker": "player"}),
    ]).dropna().drop_duplicates()
    appearances = appearances.groupby("player")["match_id"].nunique().rename("appearances").reset_index()

    prof = players.merge(batting, on="player", how="left").merge(bowling, on="player", how="left").merge(appearances, on="player", how="left")
    num_cols = prof.select_dtypes(include=np.number).columns
    prof[num_cols] = prof[num_cols].fillna(0)

    # Role signals
    prof["batting_activity"] = _safe_div(prof["balls_faced"], prof["appearances"] * 120)
    prof["bowling_activity"] = _safe_div(prof["balls_bowled"], prof["appearances"] * 120)
    prof["bowling_wicket_rate"] = _safe_div(prof["wickets"], prof["bowl_matches"])
    prof["is_bowling_specialist"] = ((prof["balls_bowled"] >= 180) & (prof["runs"] < prof["appearances"] * 22)).astype(int)
    prof["is_batting_specialist"] = ((prof["runs"] >= 200) & (prof["balls_bowled"] < 180)).astype(int)
    prof["is_all_rounder"] = ((prof["balls_faced"] >= 180) & (prof["balls_bowled"] >= 180)).astype(int)

    # Phase features
    for phase in ["Powerplay", "Middle", "Death"]:
        pbat = bat[bat["phase"].eq(phase)].copy()
        pbowl = bowl[bowl["phase"].eq(phase)].copy()
        if len(pbat):
            agg = pbat.assign(bf=(~pbat["extras_type"].eq("wides")).astype(int)).groupby("batter").agg(
                **{f"{phase.lower()}_runs": ("batsman_runs", "sum"),
                   f"{phase.lower()}_balls": ("bf", "sum"),
                   f"{phase.lower()}_sr": ("batsman_runs", "sum")}
            )
            # recompute SR denominator safely
            agg[f"{phase.lower()}_sr"] = 100 * _safe_div(agg[f"{phase.lower()}_runs"], agg[f"{phase.lower()}_balls"])
            prof = prof.merge(agg, left_on="player", right_index=True, how="left")
        if len(pbowl):
            agg2 = pbowl.assign(lb=(~pbowl["extras_type"].isin(["wides", "noballs"])).astype(int)).groupby("bowler").agg(
                **{f"{phase.lower()}_wickets": ("credited_wicket", "sum"),
                   f"{phase.lower()}_balls": ("lb", "sum"),
                   f"{phase.lower()}_runs_conceded": ("bowler_runs", "sum")}
            )
            agg2[f"{phase.lower()}_economy"] = _safe_div(agg2[f"{phase.lower()}_runs_conceded"] * 6, agg2[f"{phase.lower()}_balls"])
            prof = prof.merge(agg2, left_on="player", right_index=True, how="left")

    num_cols = prof.select_dtypes(include=np.number).columns
    prof[num_cols] = prof[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Consistency from match-level batting and bowling distributions.
    bat_match = bat.groupby(["batter", "match_id"])["batsman_runs"].sum()
    bat_std = bat_match.groupby(level=0).std().rename("runs_std")
    bowl_match = bowl.groupby(["bowler", "match_id"]).agg(
        runs=("bowler_runs", "sum"), balls=("legal_ball", "sum")
    )
    bowl_match["economy"] = _safe_div(bowl_match["runs"] * 6, bowl_match["balls"])
    econ_std = bowl_match.groupby(level=0)["economy"].std().rename("economy_std")
    prof = prof.merge(bat_std, left_on="player", right_index=True, how="left").merge(econ_std, left_on="player", right_index=True, how="left")
    prof[["runs_std","economy_std"]] = prof[["runs_std","economy_std"]].fillna(0)

    # Recent form: weighted by most recent performances, normalized to 0-100 later.
    season_rank = {s:i for i,s in enumerate(m["season"].drop_duplicates().tolist())}
    d["season_idx"] = d["season"].map(season_rank).fillna(0)
    # Recreate bowling helper columns on the season-indexed frame.
    d["legal_ball"] = (~d["extras_type"].isin(["wides", "noballs"])).astype(int)
    d["bowler_runs"] = d["total_runs"] - d["extras_type"].isin(["byes", "legbyes", "penalty"]).astype(int) * d["total_runs"]
    d["bowler_runs"] = d["bowler_runs"].clip(lower=0)
    d["credited_wicket"] = (
        d["is_wicket"].eq(1) & d["dismissal_kind"].isin(WICKET_TYPES_CREDITED_TO_BOWLER)
    ).astype(int)
    # Match-level batting and bowling impact
    bm = d[d["batter"].notna()].groupby(["batter","match_id","season_idx"])["batsman_runs"].sum().reset_index()
    wm = d[d["bowler"].notna()].groupby(["bowler","match_id","season_idx"]).agg(
        wickets=("credited_wicket","sum"), runs=("bowler_runs","sum"), balls=("legal_ball","sum")
    ).reset_index()
    wm["economy"] = _safe_div(wm["runs"]*6, wm["balls"])
    # Per-match standardized-ish signals; robustly scaled via rank later.
    br = bm.groupby("batter")["batsman_runs"].mean().rename("avg_recent_bat")
    wr = wm.groupby("bowler")["wickets"].mean().rename("avg_recent_wkts")
    prof = prof.merge(br, left_on="player", right_index=True, how="left").merge(wr, left_on="player", right_index=True, how="left")
    prof[["avg_recent_bat","avg_recent_wkts"]] = prof[["avg_recent_bat","avg_recent_wkts"]].fillna(0)

    # Role classification uses transparent thresholds based on workload and behavior.
    def role(row):
        bat_balls, bowl_balls, runs, wkts, app = row["balls_faced"], row["balls_bowled"], row["runs"], row["wickets"], max(row["appearances"],1)
        bat_per_app = bat_balls / app
        bowl_per_app = bowl_balls / app
        sr = row["strike_rate"]
        if bowl_balls >= 300 and wkts >= 8 and bat_balls >= 500:
            return "Spin/Fast All-Rounder"
        if bat_balls >= 350 and bowl_balls >= 240 and wkts >= 8:
            return "All-Rounder"
        if bowl_balls >= 360 and bowl_per_app >= 12:
            return "Bowling Specialist"
        if bat_balls >= 420:
            if sr >= 145:
                return "Finisher"
            if bat_per_app >= 25:
                return "Top/Middle-Order Batter"
            return "Batter"
        if bowl_balls >= 180:
            return "Bowling Part-Time / All-Rounder"
        return "Low-Data Player"

    prof["role"] = prof.apply(role, axis=1)
    return prof

def add_recency_scores(player_features: pd.DataFrame, deliveries: pd.DataFrame, matches: pd.DataFrame, window_type="Last 2 seasons", recent_weight=0.6):
    """Return features with 0-100 recency score. Window is season based because match-level window is not known until interactions are built."""
    p = player_features.copy()
    seasons = list(matches["season"].drop_duplicates())
    n = {"Last season": 1, "Last 2 seasons": 2, "Career": len(seasons)}.get(window_type, None)
    if n is None:
        n = 2
    recent = seasons[-n:]
    d = deliveries.merge(matches[["id","season"]], left_on="match_id", right_on="id", how="left")
    bat = d[d["batter"].notna() & d["season"].isin(recent)].groupby("batter")["batsman_runs"].sum()
    bowl = d[d["bowler"].notna() & d["season"].isin(recent)].groupby("bowler").agg(
        wkts=("is_wicket", "sum"), balls=("ball", "count")
    )
    scores = pd.DataFrame({"recent_runs": bat, "recent_wkts_raw": bowl["wkts"]}).fillna(0)
    scores["recent_component"] = 0.6 * scores["recent_runs"].rank(pct=True) * 100 + 0.4 * scores["recent_wkts_raw"].rank(pct=True) * 100
    p = p.merge(scores[["recent_component"]], left_on="player", right_index=True, how="left")
    p["recent_form_score"] = p["recent_component"].fillna(0)
    return p

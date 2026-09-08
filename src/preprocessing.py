from __future__ import annotations
import pandas as pd
import numpy as np
from .config import TEAM_ALIASES

def clean_data(deliveries: pd.DataFrame, matches: pd.DataFrame):
    d = deliveries.copy()
    m = matches.copy()

    d = d.drop_duplicates()
    m = m.drop_duplicates()

    # Types
    d["match_id"] = pd.to_numeric(d["match_id"], errors="coerce").astype("Int64")
    d["inning"] = pd.to_numeric(d["inning"], errors="coerce").astype("Int64")
    d["over"] = pd.to_numeric(d["over"], errors="coerce").astype("Int64")
    d["ball"] = pd.to_numeric(d["ball"], errors="coerce")
    for c in ["batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    m["id"] = pd.to_numeric(m["id"], errors="coerce").astype("Int64")
    m["date"] = pd.to_datetime(m["date"], errors="coerce")

    # Remove rows with no player identity where a player is required.
    d = d[d["match_id"].notna() & d["batting_team"].notna()]
    d["extras_type"] = d["extras_type"].fillna("")
    d["team_batting"] = d["batting_team"].replace(TEAM_ALIASES)
    d["team_bowling"] = d["bowling_team"].replace(TEAM_ALIASES)

    for c in ["team1", "team2", "winner", "toss_winner"]:
        if c in m.columns:
            m[c] = m[c].replace(TEAM_ALIASES)

    # Guard against impossible negative numeric values.
    for c in ["batsman_runs", "extra_runs", "total_runs"]:
        d[c] = d[c].clip(lower=0)
    d["is_wicket"] = d["is_wicket"].clip(lower=0, upper=1)

    return d, m

def dataset_summary(deliveries: pd.DataFrame, matches: pd.DataFrame) -> dict:
    teams = set(matches["team1"].dropna()) | set(matches["team2"].dropna())
    players = set(deliveries["batter"].dropna()) | set(deliveries["bowler"].dropna()) | set(deliveries["non_striker"].dropna())
    return {
        "matches": len(matches),
        "deliveries": len(deliveries),
        "seasons": matches["season"].nunique(),
        "teams": len(teams),
        "players": len(players),
    }

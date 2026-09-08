"""Central configuration for AuctionIQ."""
from dataclasses import dataclass

PHASE_BOUNDS = {
    "Powerplay": range(0, 6),
    "Middle": range(6, 15),
    "Death": range(15, 20),
}

WICKET_TYPES_CREDITED_TO_BOWLER = {
    "bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"
}

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Pune Warriors": "Pune Warriors",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

DEFAULT_WEIGHTS = {
    "content": 0.30,
    "requirement": 0.30,
    "recent_form": 0.15,
    "consistency": 0.10,
    "similarity": 0.10,
    "preference": 0.05,
}

RISK_TARGETS = {
    "Low": {"data": 0.85, "consistency": 0.75},
    "Medium": {"data": 0.65, "consistency": 0.55},
    "High": {"data": 0.45, "consistency": 0.35},
}

RECENCY_WINDOWS = {
    "Last 5 matches": 5,
    "Last 10 matches": 10,
    "Last season": 1,
    "Last 2 seasons": 2,
    "Career": None,
}

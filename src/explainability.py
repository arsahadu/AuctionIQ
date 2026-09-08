from __future__ import annotations
import pandas as pd

def explain_recommendation(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("role_match", 0) >= 80:
        reasons.append(f"Role match is strong ({row['role_match']:.0f}/100).")
    if row.get("batting_strength", 0) >= 70:
        reasons.append(f"Batting profile is strong ({row['batting_strength']:.0f}/100).")
    if row.get("bowling_strength", 0) >= 70:
        reasons.append(f"Bowling profile is strong ({row['bowling_strength']:.0f}/100).")
    if row.get("recent_form_score", 0) >= 70:
        reasons.append(f"Recent-form signal is high ({row['recent_form_score']:.0f}/100).")
    if row.get("consistency_score", 0) >= 70:
        reasons.append(f"Consistency is high ({row['consistency_score']:.0f}/100).")
    if row.get("death_score", 0) >= 70:
        reasons.append("Strong death-over contribution.")
    if row.get("experience_score", 0) >= 70:
        reasons.append("Substantial historical IPL experience.")
    if not reasons:
        reasons.append("Recommended because the combined personalized score is relatively strong for the selected requirements.")
    return reasons

def confidence_label(row):
    return row.get("confidence", "Low")

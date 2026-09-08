from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

CONTENT_FEATURES = [
    "batting_strength", "bowling_strength", "recent_form_score", "consistency_score",
    "powerplay_score", "middle_score", "death_score", "experience_score"
]

def fit_content_model(player_profiles: pd.DataFrame):
    X = player_profiles[CONTENT_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sim = cosine_similarity(Xs)
    return scaler, Xs, sim, CONTENT_FEATURES

def similar_players(player_profiles: pd.DataFrame, similarity_matrix, player: str, top_k: int = 5):
    if player not in set(player_profiles["player"]):
        return pd.DataFrame(columns=["player","similarity"])
    idx = player_profiles.index[player_profiles["player"].eq(player)][0]
    sims = similarity_matrix[idx]
    candidates = player_profiles[["player","role"]].copy()
    candidates["similarity"] = sims
    candidates = candidates[candidates["player"] != player].sort_values("similarity", ascending=False).head(top_k)
    candidates["similarity"] = 100 * candidates["similarity"].clip(lower=0)
    return candidates.reset_index(drop=True)

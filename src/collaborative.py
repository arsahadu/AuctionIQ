from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def build_team_player_matrix(deliveries: pd.DataFrame, matches: pd.DataFrame):
    d = deliveries.copy()
    m = matches[["id","team1","team2"]].copy()
    # Players are observed through batting or bowling delivery participation.
    long_parts = [
        d[["match_id","batting_team","batter"]].rename(columns={"batting_team":"team","batter":"player"}),
        d[["match_id","bowling_team","bowler"]].rename(columns={"bowling_team":"team","bowler":"player"})
    ]
    interactions = pd.concat(long_parts).dropna().drop_duplicates()
    # Historical franchise aliases are already normalized in preprocessing.
    matrix = interactions.assign(value=1).pivot_table(index="team", columns="player", values="value", aggfunc="sum", fill_value=0)
    return matrix

def fit_implicit_team_model(team_player_matrix: pd.DataFrame, n_components: int = 8):
    if team_player_matrix.shape[0] < 3 or team_player_matrix.shape[1] < 3:
        return None
    n = min(n_components, team_player_matrix.shape[0]-1, team_player_matrix.shape[1]-1)
    if n < 2:
        return None
    svd = TruncatedSVD(n_components=n, random_state=42)
    team_latent = svd.fit_transform(team_player_matrix)
    player_latent = svd.components_.T
    player_latent = normalize(player_latent)
    return svd, team_latent, player_latent

def team_affinity_scores(team_player_matrix, model_output, team_name: str, player_names):
    if model_output is None or team_name not in team_player_matrix.index:
        return pd.Series(0.0, index=player_names)
    svd, team_latent, player_latent = model_output
    t_idx = team_player_matrix.index.get_loc(team_name)
    query = team_latent[t_idx].reshape(1,-1)
    scores = (normalize(query) @ player_latent.T).ravel()
    return pd.Series(scores, index=team_player_matrix.columns).reindex(player_names).fillna(0)

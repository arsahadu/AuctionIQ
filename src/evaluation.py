from __future__ import annotations
import numpy as np
import pandas as pd

def precision_at_k(recommended, relevant, k):
    rec = list(recommended)[:k]
    return sum(x in set(relevant) for x in rec) / max(k,1)

def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return np.nan
    rec = list(recommended)[:k]
    return sum(x in relevant for x in rec) / len(relevant)

def hit_rate_at_k(recommended, relevant, k):
    relevant = set(relevant)
    return float(any(x in relevant for x in list(recommended)[:k])) if relevant else np.nan

def ndcg_at_k(recommended, relevant, k):
    rel = set(relevant)
    rec = list(recommended)[:k]
    gains = [1 if x in rel else 0 for x in rec]
    dcg = sum(g / np.log2(i+2) for i,g in enumerate(gains))
    ideal = min(len(rel), k)
    idcg = sum(1 / np.log2(i+2) for i in range(ideal))
    return dcg/idcg if idcg else np.nan

def catalog_coverage(recommended_lists, catalog_size):
    unique = set()
    for recs in recommended_lists:
        unique.update(recs)
    return len(unique) / max(catalog_size,1)

def intra_list_diversity(recommendation_df, feature_cols):
    if len(recommendation_df) < 2:
        return 0.0
    X = recommendation_df[feature_cols].fillna(0).to_numpy(dtype=float)
    sims = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            a,b=X[i],X[j]
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            sims.append(float(a.dot(b)/denom) if denom else 0)
    return 1 - float(np.mean(sims)) if sims else 0.0

def evaluate_against_team_history(
    recs, target_team_history, k=5
):
    # This is an optional, proxy offline test: future/recent team participation as relevance.
    relevant = set(target_team_history)
    return {
        "Precision@K": precision_at_k(recs, relevant, k),
        "Recall@K": recall_at_k(recs, relevant, k),
        "Hit Rate@K": hit_rate_at_k(recs, relevant, k),
        "NDCG@K": ndcg_at_k(recs, relevant, k),
    }

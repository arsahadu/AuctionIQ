"""Similarity helpers kept separate for academic demonstration."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_player_similarity(X):
    return cosine_similarity(X)

def vector_similarity(vec_a, vec_b):
    a = np.asarray(vec_a).reshape(1,-1)
    b = np.asarray(vec_b).reshape(1,-1)
    return float(cosine_similarity(a,b)[0,0])

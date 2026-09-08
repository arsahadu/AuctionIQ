from __future__ import annotations
import pandas as pd

from .content_based import fit_content_model
from .ranking import calculate_recommendations

class AuctionIQRecommender:
    def __init__(self, profiles: pd.DataFrame):
        self.profiles = profiles.copy()
        self.scaler, self.X_scaled, self.similarity_matrix, self.content_features = fit_content_model(self.profiles)

    def similar(self, player, top_k=5):
        from .content_based import similar_players
        return similar_players(self.profiles, self.similarity_matrix, player, top_k)

    def recommend(
        self,
        required_role="Any",
        batting_importance=0.5,
        bowling_importance=0.5,
        recent_importance=0.5,
        consistency_importance=0.5,
        risk="Medium",
        top_k=5,
        weights=None,
    ):
        # No fake user ratings are injected. Team requirements are the personalization vector.
        sim_seed = pd.Series(0.0, index=self.profiles["player"])
        return calculate_recommendations(
            self.profiles,
            required_role=required_role,
            batting_importance=batting_importance,
            bowling_importance=bowling_importance,
            recent_importance=recent_importance,
            consistency_importance=consistency_importance,
            risk=risk,
            top_k=top_k,
            weights=weights,
            similarity_seed=sim_seed,
        )

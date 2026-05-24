"""
hybrid_recommender.py
=====================
Hybrid Recommendation Engine

Combines Content-Based Filtering (TF-IDF cosine similarity) and
Collaborative Filtering (SVD latent factors) using a weighted ensemble.

Strategy
--------
  hybrid_score = α × cb_score + β × cf_score + γ × popularity_score

  α, β, γ are tunable weights (default 0.5, 0.3, 0.2).

Also provides a KNN-based genre recommender as a lightweight alternative.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


# ══════════════════════════════════════════════════════════════════════════════
class HybridRecommender:
    """
    Hybrid recommender: Content-Based + Collaborative Filtering.

    Parameters
    ----------
    alpha : weight for content-based similarity
    beta  : weight for collaborative-filter score
    gamma : weight for popularity (log members)
    """

    def __init__(
        self,
        alpha: float = 0.50,
        beta:  float = 0.30,
        gamma: float = 0.20,
        model_path: str = "hybrid_model.pkl",
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.model_path = model_path

        self.cb_rec  : ContentBasedRecommender | None = None
        self.cf_rec  : CollaborativeFilteringRecommender | None = None
        self.df      : pd.DataFrame | None = None
        self._scaler : MinMaxScaler | None = None

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        ratings_df: pd.DataFrame | None = None,
        cb_max_features: int = 15_000,
        cf_n_factors: int = 50,
    ) -> "HybridRecommender":
        """
        Fit both sub-models.

        Parameters
        ----------
        df              : cleaned anime metadata
        ratings_df      : optional real user ratings; synthetic if None
        cb_max_features : vocab size for TF-IDF
        cf_n_factors    : SVD latent dimensions
        """
        self.df = df.reset_index(drop=True)

        print("[Hybrid] fitting Content-Based model ...")
        self.cb_rec = ContentBasedRecommender(
            max_features=cb_max_features, model_path=None
        ).fit(df)

        print("[Hybrid] fitting Collaborative Filtering model ...")
        self.cf_rec = CollaborativeFilteringRecommender(
            n_factors=cf_n_factors, model_path=None
        ).fit(df, ratings_df)

        # scaler for popularity normalisation
        self._scaler = MinMaxScaler()
        self._scaler.fit(self.df[["popularity_score"]])

        if self.model_path:
            self._save()

        return self

    # ── persistence ───────────────────────────────────────────────────────────
    def _save(self) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Hybrid] model saved -> {self.model_path}")

    @classmethod
    def load(cls, model_path: str = "hybrid_model.pkl") -> "HybridRecommender":
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        print(f"[Hybrid] model loaded <- {model_path}")
        return obj

    # ── recommendation ────────────────────────────────────────────────────────
    def recommend(
        self,
        title: str,
        n: int = 10,
        user_id: int | None = None,
        filter_type: str | None = None,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """
        Hybrid recommendations for a given anime title.

        Parameters
        ----------
        title       : anime name to base recommendations on
        n           : number of results
        user_id     : if given, personalise with that user's CF scores
        filter_type : restrict results to a specific anime type ('TV', 'Movie', …)
        min_score   : minimum MAL score filter

        Returns
        -------
        DataFrame sorted by hybrid_score descending
        """
        # ── content-based scores (top 200 candidates) ─────────────────────
        cb_results = self.cb_rec.recommend(title, n=200, filter_type=filter_type)
        # normalise cb similarity to [0, 1] (already is, but just in case)
        cb_results["cb_norm"] = cb_results["similarity"].clip(0, 1)

        # ── collaborative-filter scores ────────────────────────────────────
        if user_id is not None:
            try:
                pred_ratings = self.cf_rec.predict_user_ratings(user_id)
                # normalise to [0, 1]
                r_min, r_max = pred_ratings.min(), pred_ratings.max()
                pred_ratings_norm = (pred_ratings - r_min) / (r_max - r_min + 1e-9)

                def _cf_score(anime_name):
                    row = self.df[self.df["Name"] == anime_name]
                    if row.empty:
                        return 0.0
                    aid = row.iloc[0]["anime_id"]
                    return float(pred_ratings_norm.get(aid, 0.0))

                cb_results["cf_norm"] = cb_results["Name"].apply(_cf_score)
            except Exception:
                cb_results["cf_norm"] = 0.0
        else:
            cb_results["cf_norm"] = 0.0

        # ── popularity score ───────────────────────────────────────────────
        def _pop_score(anime_name):
            row = self.df[self.df["Name"] == anime_name]
            if row.empty:
                return 0.0
            val = row.iloc[0]["popularity_score"]
            return float(self._scaler.transform([[val]])[0][0])

        cb_results["pop_norm"] = cb_results["Name"].apply(_pop_score)

        # ── combined score ────────────────────────────────────────────────
        cb_results["hybrid_score"] = (
            self.alpha * cb_results["cb_norm"] +
            self.beta  * cb_results["cf_norm"] +
            self.gamma * cb_results["pop_norm"]
        ).round(4)

        # filter by min score
        if min_score > 0:
            cb_results = cb_results[cb_results["Score"] >= min_score]

        result = (
            cb_results
            .sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        return result[["Name", "Score", "Genres", "Type", "Episodes",
                        "cb_norm", "cf_norm", "pop_norm", "hybrid_score"]]

    # ── KNN genre recommender ─────────────────────────────────────────────────
    def knn_genre_recommend(
        self,
        genres: list[str],
        n: int = 10,
        min_score: float = 7.0,
    ) -> pd.DataFrame:
        """
        KNN-based recommender: find anime closest to a user-defined
        genre + feature vector using Euclidean distance on a numeric feature matrix.

        Feature matrix columns:
            - One-hot encoded top genres
            - Normalised Score
            - Normalised Popularity
        """
        top_genres = [
            "Action", "Adventure", "Comedy", "Drama", "Fantasy",
            "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life",
            "Sports", "Supernatural", "Thriller", "Psychological",
            "Mecha", "Music", "School", "Shounen", "Seinen",
        ]

        # build feature matrix for all anime
        df = self.df.copy()
        for g in top_genres:
            df[f"g_{g}"] = df["genre_list"].apply(lambda gl: int(g in gl))

        scaler = MinMaxScaler()
        df["score_norm"] = scaler.fit_transform(df[["Score"]])
        df["pop_norm2"]  = scaler.fit_transform(df[["popularity_score"]])

        feat_cols = [f"g_{g}" for g in top_genres] + ["score_norm", "pop_norm2"]
        X = df[feat_cols].values.astype(np.float32)

        # build query vector from desired genres
        query = np.zeros(len(feat_cols), dtype=np.float32)
        for i, g in enumerate(top_genres):
            if g in genres:
                query[i] = 1.0
        query[-2] = 1.0   # prefer high score
        query[-1] = 0.5   # moderate popularity

        knn = NearestNeighbors(n_neighbors=n + 1, metric="euclidean")
        knn.fit(X)
        distances, indices = knn.kneighbors([query])

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = df.iloc[idx]
            if row["Score"] < min_score:
                continue
            results.append({
                "Name":     row["Name"],
                "Score":    row["Score"],
                "Genres":   row["Genres"],
                "Type":     row["Type"],
                "Episodes": row["Episodes"],
                "knn_dist": round(float(dist), 4),
            })

        return pd.DataFrame(results).head(n).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_preprocessing import load_raw_data, preprocess

    df = preprocess(load_raw_data())

    if os.path.exists("hybrid_model.pkl"):
        rec = HybridRecommender.load()
    else:
        rec = HybridRecommender(alpha=0.5, beta=0.3, gamma=0.2).fit(df)

    print("\n-- Hybrid recs for 'Fullmetal Alchemist: Brotherhood' -----------")
    print(rec.recommend("Fullmetal Alchemist: Brotherhood", n=10).to_string(index=False))

    print("\n-- Hybrid recs with user_id=42 ----------------------------------")
    print(rec.recommend("Fullmetal Alchemist: Brotherhood", n=10, user_id=42).to_string(index=False))

    print("\n-- KNN genre recs: Action + Fantasy -----------------------------")
    print(rec.knn_genre_recommend(["Action", "Fantasy"], n=10).to_string(index=False))

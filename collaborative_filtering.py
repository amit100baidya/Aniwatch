"""
collaborative_filtering.py
==========================
Collaborative Filtering via SVD (Singular Value Decomposition).

Because this dataset has no user-rating rows, we SIMULATE a synthetic
user-item rating matrix from:
  - Score       (proxy for community rating)
  - Members     (proxy for audience size)
  - Favorites   (proxy for strong preference)

This makes the project self-contained.  In production you would replace
`build_synthetic_ratings()` with a real user×anime ratings CSV.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


# ══════════════════════════════════════════════════════════════════════════════
def build_synthetic_ratings(
    df: pd.DataFrame,
    n_users: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a user × anime rating matrix.

    Each 'user' has a random genre preference vector.  An anime is
    'rated' by a user if its top genres align with that user's preferences.
    The rating value is derived from the anime's Score + noise.

    Returns a DataFrame with columns: user_id, anime_id, rating (1-10).
    """
    rng = np.random.default_rng(seed)
    print(f"[CF] building synthetic ratings for {n_users} users ...")

    # collect all genres
    all_genres = set()
    for gl in df["genre_list"]:
        all_genres.update(gl)
    all_genres = sorted(all_genres)
    n_genres = len(all_genres)
    genre_idx = {g: i for i, g in enumerate(all_genres)}

    # genre vector per anime (binary)
    anime_genre_vec = np.zeros((len(df), n_genres), dtype=np.float32)
    for row_i, gl in enumerate(df["genre_list"]):
        for g in gl:
            if g in genre_idx:
                anime_genre_vec[row_i, genre_idx[g]] = 1.0

    # genre preference per user (random Dirichlet)
    user_genre_pref = rng.dirichlet(np.ones(n_genres) * 0.5, size=n_users)

    # affinity score = user_genre_pref · anime_genre_vec  (n_users × n_anime)
    affinity = user_genre_pref @ anime_genre_vec.T      # (2000, n_anime)

    # normalised anime score (0-1)
    scores_norm = (df["Score"].values - df["Score"].min()) / (
        df["Score"].max() - df["Score"].min() + 1e-9
    )

    # combined signal: 60 % affinity + 40 % score
    combined = 0.6 * affinity + 0.4 * scores_norm[np.newaxis, :]

    # threshold: user 'rates' top-k anime per user (sparse)
    k = 150
    records = []
    anime_ids = df["anime_id"].values
    for u in range(n_users):
        top_k = np.argpartition(combined[u], -k)[-k:]
        for a_idx in top_k:
            raw_rating = combined[u, a_idx]
            # map to 1-10
            rating = float(np.clip(raw_rating * 10, 1, 10))
            # add small noise
            rating = float(np.clip(rating + rng.normal(0, 0.5), 1, 10))
            records.append((u, int(anime_ids[a_idx]), round(rating, 1)))

    ratings_df = pd.DataFrame(records, columns=["user_id", "anime_id", "rating"])
    print(f"[CF] synthetic ratings: {len(ratings_df):,} rows")
    return ratings_df


# ══════════════════════════════════════════════════════════════════════════════
class CollaborativeFilteringRecommender:
    """
    SVD-based collaborative filter.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to keep after SVD.
    """

    def __init__(self, n_factors: int = 50, model_path: str = "cf_model.pkl"):
        self.n_factors  = n_factors
        self.model_path = model_path

        # set after fit()
        self.user_factors  = None
        self.sigma         = None
        self.anime_factors = None
        self.user_item_df  = None   # dense pivot (user × anime)
        self.df            = None   # anime metadata
        self._anime_id_to_idx  = None
        self._anime_idx_to_row = None

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        ratings_df: pd.DataFrame | None = None,
    ) -> "CollaborativeFilteringRecommender":
        """
        Parameters
        ----------
        df         : cleaned anime metadata DataFrame
        ratings_df : user×anime ratings (user_id, anime_id, rating).
                     If None, synthetic ratings are generated automatically.
        """
        self.df = df.reset_index(drop=True)

        if ratings_df is None:
            ratings_df = build_synthetic_ratings(df)

        print("[CF] building user-item matrix ...")
        # pivot: rows = users, columns = anime_id
        self.user_item_df = ratings_df.pivot_table(
            index="user_id", columns="anime_id", values="rating"
        ).fillna(0)

        # mean-centre per user (subtract row mean)
        user_means = self.user_item_df.mean(axis=1)
        matrix_centered = self.user_item_df.sub(user_means, axis=0)

        # SVD
        print(f"[CF] running SVD with {self.n_factors} factors ...")
        sparse_mat = csr_matrix(matrix_centered.values)
        U, sigma, Vt = svds(sparse_mat, k=self.n_factors)

        # sort by singular value (svds returns ascending)
        order = np.argsort(sigma)[::-1]
        self.user_factors  = U[:, order]
        self.sigma         = sigma[order]
        self.anime_factors = Vt[order, :]   # shape (n_factors, n_anime)

        self._anime_id_to_idx  = {aid: i for i, aid in enumerate(self.user_item_df.columns)}
        self._anime_idx_to_row = {i: aid for i, aid in enumerate(self.user_item_df.columns)}

        print("[CF] SVD done")
        if self.model_path:
            self._save()
        return self

    # ── persistence ───────────────────────────────────────────────────────────
    def _save(self) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[CF] model saved -> {self.model_path}")

    @classmethod
    def load(cls, model_path: str = "cf_model.pkl") -> "CollaborativeFilteringRecommender":
        obj = cls(model_path=model_path)
        with open(model_path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        print(f"[CF] model loaded <- {model_path}")
        return obj

    # ── prediction ────────────────────────────────────────────────────────────
    def predict_user_ratings(self, user_id: int) -> pd.Series:
        """
        Predict ratings for all anime for a given user_id.
        Returns a Series indexed by anime_id.
        """
        if user_id not in range(len(self.user_factors)):
            raise ValueError(f"user_id {user_id} out of range [0, {len(self.user_factors)-1}]")

        u_vec = self.user_factors[user_id] * self.sigma          # (n_factors,)
        pred  = u_vec @ self.anime_factors                        # (n_anime,)
        return pd.Series(pred, index=self.user_item_df.columns)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Top-N recommendations for an existing user.

        Parameters
        ----------
        user_id      : integer user id
        n            : number to return
        exclude_seen : if True, exclude anime the user already rated
        """
        pred = self.predict_user_ratings(user_id)

        if exclude_seen:
            seen_ids = set(
                self.user_item_df.columns[self.user_item_df.iloc[user_id] > 0]
            )
            pred = pred.drop(index=list(seen_ids & set(pred.index)), errors="ignore")

        top_ids = pred.nlargest(n).index.tolist()
        results = []
        for aid in top_ids:
            row = self.df[self.df["anime_id"] == aid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "Name":            row["Name"],
                "Score":           row["Score"],
                "Genres":          row["Genres"],
                "Type":            row["Type"],
                "predicted_rating": round(pred[aid], 2),
            })

        return pd.DataFrame(results)

    # ── item-item similarity via latent factors ────────────────────────────────
    def similar_anime(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Find anime with similar latent factor vectors to *title*.
        """
        row = self.df[self.df["Name"].str.lower() == title.lower()]
        if row.empty:
            # partial match
            row = self.df[self.df["Name"].str.lower().str.contains(title.lower())]
            if row.empty:
                raise ValueError(f"Anime '{title}' not found.")
        anime_id = row.iloc[0]["anime_id"]

        if anime_id not in self._anime_id_to_idx:
            raise ValueError(f"'{title}' not in user-item matrix (no ratings).")

        idx      = self._anime_id_to_idx[anime_id]
        item_vec = self.anime_factors[:, idx]               # (n_factors,)

        # cosine sim with all items
        norms = np.linalg.norm(self.anime_factors, axis=0)
        sims  = (item_vec @ self.anime_factors) / (np.linalg.norm(item_vec) * norms + 1e-9)

        top_idx = np.argsort(sims)[::-1]
        results = []
        for i in top_idx:
            aid = self._anime_idx_to_row[i]
            if aid == anime_id:
                continue
            meta = self.df[self.df["anime_id"] == aid]
            if meta.empty:
                continue
            meta = meta.iloc[0]
            results.append({
                "Name":       meta["Name"],
                "Score":      meta["Score"],
                "Genres":     meta["Genres"],
                "Type":       meta["Type"],
                "cf_similarity": round(float(sims[i]), 4),
            })
            if len(results) == n:
                break

        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_preprocessing import load_raw_data, preprocess

    df = preprocess(load_raw_data())

    if os.path.exists("cf_model.pkl"):
        rec = CollaborativeFilteringRecommender.load()
    else:
        rec = CollaborativeFilteringRecommender(n_factors=50).fit(df)

    print("\n-- Top-10 for user 0 --------------------------------------------")
    print(rec.recommend_for_user(0, n=10).to_string(index=False))

    print("\n-- Anime similar to 'Attack on Titan' (CF) ----------------------")
    print(rec.similar_anime("Attack on Titan", n=10).to_string(index=False))

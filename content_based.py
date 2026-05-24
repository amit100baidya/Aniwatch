"""
content_based.py
================
Content-Based Filtering using TF-IDF on text features
(Genres, Type, Studios, Source, Rating, Synopsis).

How it works
------------
1. Build a TF-IDF matrix from combined text features.
2. Compute cosine similarity between all anime pairs (done lazily per query).
3. Given an anime title, return the N most-similar anime.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


# ══════════════════════════════════════════════════════════════════════════════
class ContentBasedRecommender:
    """
    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for TF-IDF (default 15 000).
    model_path : str | None
        If given, the fitted vectoriser + matrix are cached here.
    """

    def __init__(self, max_features: int = 15_000, model_path: str = "cb_model.pkl"):
        self.max_features = max_features
        self.model_path   = model_path
        self.vectorizer   = None
        self.tfidf_matrix = None
        self.df           = None     # reference to the cleaned dataframe
        self._title_index = None    # anime Name → row index

    # ── fitting ───────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build the TF-IDF matrix from df['text_features'].

        Parameters
        ----------
        df : cleaned DataFrame produced by data_preprocessing.preprocess()
        """
        print("[ContentBased] fitting TF-IDF ...")
        self.df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),          # unigrams + bigrams
            sublinear_tf=True,           # log-norm TF
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["text_features"].fillna("")
        )

        # map lowercase title → row index for fast lookup
        self._title_index = pd.Series(
            self.df.index, index=self.df["Name"].str.lower()
        )
        print(f"[ContentBased] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        if self.model_path:
            self._save()

        return self

    # ── persistence ───────────────────────────────────────────────────────────
    def _save(self) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix, self.df), f)
        print(f"[ContentBased] model saved -> {self.model_path}")

    @classmethod
    def load(cls, model_path: str = "cb_model.pkl") -> "ContentBasedRecommender":
        obj = cls(model_path=model_path)
        with open(model_path, "rb") as f:
            obj.vectorizer, obj.tfidf_matrix, obj.df = pickle.load(f)
        obj._title_index = pd.Series(
            obj.df.index, index=obj.df["Name"].str.lower()
        )
        print(f"[ContentBased] model loaded <- {model_path}")
        return obj

    # ── recommendation ────────────────────────────────────────────────────────
    def recommend(
        self,
        title: str,
        n: int = 10,
        filter_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Return the top-N most-similar anime to *title*.

        Parameters
        ----------
        title       : anime name (case-insensitive)
        n           : number of recommendations
        filter_type : if provided (e.g. 'TV', 'Movie'), only return that type

        Returns
        -------
        DataFrame with columns: Name, Score, Genres, Type, Studios, similarity
        """
        key = title.lower().strip()
        if key not in self._title_index:
            # fuzzy fallback: find closest match
            matches = self._title_index.index[
                self._title_index.index.str.contains(key, regex=False)
            ]
            if len(matches) == 0:
                raise ValueError(f"Anime '{title}' not found in dataset.")
            key = matches[0]
            print(f"[ContentBased] '{title}' not found; using '{key}' instead.")

        idx = self._title_index[key]
        if isinstance(idx, pd.Series):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)

        # cosine similarity between this anime and all others
        vec  = self.tfidf_matrix[idx]
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        # sort, exclude self
        sim_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]

        # optional type filter
        if filter_type:
            sim_scores = [
                (i, s) for i, s in sim_scores
                if self.df.loc[i, "Type"].lower() == filter_type.lower()
            ]

        top = sim_scores[:n]
        rows = []
        for i, score in top:
            rows.append({
                "Name":       self.df.loc[i, "Name"],
                "Score":      self.df.loc[i, "Score"],
                "Genres":     self.df.loc[i, "Genres"],
                "Type":       self.df.loc[i, "Type"],
                "Studios":    self.df.loc[i, "Studios"],
                "Episodes":   self.df.loc[i, "Episodes"],
                "similarity": round(score, 4),
            })

        return pd.DataFrame(rows)

    # ── genre-based recommendation ────────────────────────────────────────────
    def recommend_by_genres(
        self,
        genres: list[str],
        n: int = 10,
        min_score: float = 7.0,
    ) -> pd.DataFrame:
        """
        Return top anime that contain ALL specified genres,
        sorted by Score descending.

        Parameters
        ----------
        genres    : list of genre strings e.g. ['Action', 'Sci-Fi']
        n         : number to return
        min_score : minimum MAL score threshold
        """
        genres_lower = [g.lower() for g in genres]
        mask = self.df["genre_list"].apply(
            lambda gl: all(g in [x.lower() for x in gl] for g in genres_lower)
        )
        filtered = self.df[mask & (self.df["Score"] >= min_score)].copy()
        filtered = filtered.sort_values("Score", ascending=False).head(n)
        return filtered[["Name", "Score", "Genres", "Type", "Studios", "Episodes"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_preprocessing import load_raw_data, preprocess, save_processed

    # build or load
    if os.path.exists("cb_model.pkl"):
        rec = ContentBasedRecommender.load()
    else:
        df  = preprocess(load_raw_data())
        save_processed(df)
        rec = ContentBasedRecommender().fit(df)

    print("\n-- Similar to 'Naruto' -----------------------------------------")
    print(rec.recommend("Naruto", n=10).to_string(index=False))

    print("\n-- Similar to 'Death Note' (TV only) ----------------------------")
    print(rec.recommend("Death Note", n=10, filter_type="TV").to_string(index=False))

    print("\n-- Top Action + Sci-Fi anime (score >= 8.0) ---------------------")
    print(rec.recommend_by_genres(["Action", "Sci-Fi"], n=10, min_score=8.0).to_string(index=False))

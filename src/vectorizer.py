# TF-IDF index over all candidate text, scored against the JD.
# For 100k candidates with a ~10k-token vocab this uses ~100-200 MB RAM
# and scores everything in under 3 seconds.
#
# Usage:
#   cv = CorpusVectorizer()
#   cv.fit(candidates)
#   scores = cv.score_all(jd_raw_text)  # numpy array, shape (n_candidates,)

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _candidate_text(candidate: dict) -> str:
    """Extract a single searchable string from a candidate record."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        " ".join(c.get("title", "") for c in career),
        " ".join(c.get("description", "") for c in career),
        " ".join(s.get("name", "") for s in skills),
    ]
    return " ".join(p for p in parts if p)


class CorpusVectorizer:
    """Fits a TF-IDF index on candidate docs and scores them against a query (the JD)."""

    def __init__(
        self,
        max_features: int = 15_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,     # log(1+tf) dampens repeated terms
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-\+#]*",
        )
        self._candidate_matrix = None
        self._fitted = False

    def fit(self, candidates: list[dict]) -> "CorpusVectorizer":
        """Build the TF-IDF index from a list of candidate dicts."""
        corpus = [_candidate_text(c) for c in candidates]
        self._candidate_matrix = self._vectorizer.fit_transform(corpus)
        # L2-normalise so dot product == cosine similarity
        self._candidate_matrix = normalize(self._candidate_matrix, norm="l2", copy=False)
        self._fitted = True
        return self

    def score_all(self, jd_text: str) -> np.ndarray:
        """Cosine similarity of every candidate against jd_text. Returns array of shape (n,)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score_all().")

        jd_vec = self._vectorizer.transform([jd_text])   # sparse (1, vocab)
        jd_vec = normalize(jd_vec, norm="l2", copy=False)

        # sparse dot product: (n_candidates, vocab) @ (vocab, 1) -> (n_candidates, 1)
        scores = (self._candidate_matrix @ jd_vec.T).toarray().ravel()
        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    def score_subset(self, indices: list[int], jd_text: str) -> np.ndarray:
        """
        Score only a subset of candidates (e.g. the Stage-2 shortlist).
        Returns array of length len(indices).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score_subset().")

        jd_vec = self._vectorizer.transform([jd_text])
        jd_vec = normalize(jd_vec, norm="l2", copy=False)

        sub_matrix = self._candidate_matrix[indices]     # sparse slice
        scores = (sub_matrix @ jd_vec.T).toarray().ravel()
        return scores.astype(np.float32)

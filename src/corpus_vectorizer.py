"""
Corpus-wide TF-IDF vectorizer for fast candidate-to-JD text similarity.

Replaces simple_bm25_score() in matcher.py for 100k-scale use.
Builds an sklearn TfidfVectorizer index over all candidate texts once, then
scores every candidate against the JD via a single sparse matrix dot product —
O(vocab × n_docs) total, not O(n_docs × |jd_vocab|) per doc.

Usage:
    from corpus_vectorizer import CorpusVectorizer
    cv = CorpusVectorizer()
    cv.fit(candidates)                      # ~30s for 100k docs
    scores = cv.score_all(jd_raw_text)      # numpy array, shape (n_candidates,)
"""

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
    """
    Fits a TF-IDF index on a corpus of candidates and scores them against a
    query document (the JD) using cosine similarity.

    The matrix is kept sparse throughout — for 100k candidates with a typical
    10k-token vocabulary this uses ~100–200 MB of RAM and scores in <3s.
    """

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
            sublinear_tf=sublinear_tf,     # log(1+tf) — dampens repeated terms
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-\+#]*",
        )
        self._candidate_matrix = None   # sparse (n_candidates, vocab)
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, candidates: list[dict]) -> "CorpusVectorizer":
        """
        Build the TF-IDF index from a list of candidate dicts.

        Args:
            candidates: list of raw candidate dicts (same schema as matcher.py).
                        Preserves original list order so scores[i] maps to candidates[i].
        Returns:
            self  (fluent interface)
        """
        corpus = [_candidate_text(c) for c in candidates]
        self._candidate_matrix = self._vectorizer.fit_transform(corpus)  # sparse
        # L2-normalise rows so dot product == cosine similarity
        self._candidate_matrix = normalize(self._candidate_matrix, norm="l2", copy=False)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def score_all(self, jd_text: str) -> np.ndarray:
        """
        Return a 1-D numpy array of cosine similarities, shape (n_candidates,).
        Values are in [0, 1]; higher = more textually similar to the JD.

        Args:
            jd_text: raw JD text string (same as jd["raw_text_for_bm25"]).
        """
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

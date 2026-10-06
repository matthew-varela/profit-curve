"""Build dated NLP features from downloaded SEC filing text."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize

from feature_config import FILINGS_DIR, FILING_FEATURES_OUT, NLP_EMBEDDING_DIMS

UNCERTAINTY_WORDS = {
    "may",
    "might",
    "could",
    "uncertain",
    "uncertainty",
    "estimate",
    "estimated",
    "approximately",
    "contingent",
    "possible",
}
RISK_WORDS = {
    "risk",
    "risks",
    "adverse",
    "material",
    "impairment",
    "volatility",
    "exposure",
    "disruption",
    "default",
}
LITIGATION_WORDS = {
    "lawsuit",
    "litigation",
    "claim",
    "claims",
    "proceeding",
    "investigation",
    "regulatory",
    "settlement",
}


def discover_filings() -> pd.DataFrame:
    manifest = FILINGS_DIR / "manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        df["path"] = df["path"].map(Path)
        return df

    rows = []
    for path in FILINGS_DIR.glob("*/*.txt"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        rows.append(
            {
                "cik": path.parent.name,
                "form": parts[1],
                "filing_date": parts[0],
                "accession": parts[2],
                "primary_doc": path.name,
                "path": path,
            }
        )
    return pd.DataFrame(rows)


def basic_text_stats(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text.lower())
    alpha_words = [w for w in words if w.isalpha()]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = max(len(words), 1)
    alpha_count = max(len(alpha_words), 1)

    return {
        "filing_text_len_log": float(np.log1p(len(text))),
        "filing_word_count_log": float(np.log1p(len(words))),
        "filing_avg_word_len": float(np.mean([len(w) for w in alpha_words]) if alpha_words else 0.0),
        "filing_sentence_count_log": float(np.log1p(len(sentences))),
        "filing_uncertainty_rate": sum(w in UNCERTAINTY_WORDS for w in alpha_words) / alpha_count,
        "filing_risk_rate": sum(w in RISK_WORDS for w in alpha_words) / alpha_count,
        "filing_litigation_rate": sum(w in LITIGATION_WORDS for w in alpha_words) / alpha_count,
        "filing_numeric_rate": sum(any(ch.isdigit() for ch in w) for w in words) / word_count,
    }


def make_embeddings(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        raw = model.encode(texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
        raw = np.asarray(raw, dtype=np.float32)
    except Exception as exc:
        print(f"WARNING: sentence-transformers unavailable ({exc}); using hashed text embeddings.")
        vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )
        raw = vectorizer.transform(texts).toarray().astype(np.float32)

    if raw.shape[1] > NLP_EMBEDDING_DIMS:
        projector = GaussianRandomProjection(n_components=NLP_EMBEDDING_DIMS, random_state=42)
        raw = projector.fit_transform(raw).astype(np.float32)
        raw = normalize(raw)
    elif raw.shape[1] < NLP_EMBEDDING_DIMS:
        pad = np.zeros((raw.shape[0], NLP_EMBEDDING_DIMS - raw.shape[1]), dtype=np.float32)
        raw = np.hstack([raw, pad])

    return raw[:, :NLP_EMBEDDING_DIMS]


def main() -> None:
    filings = discover_filings()
    if filings.empty:
        raise RuntimeError("No filing text files found. Run sec_download.py first.")

    records = []
    texts = []
    for row in filings.itertuples(index=False):
        text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        records.append(
            {
                "cik": str(row.cik).zfill(10),
                "form": row.form,
                "filing_date": pd.to_datetime(row.filing_date, utc=True),
                **basic_text_stats(text),
            }
        )
        texts.append(text[:300_000])

    if not records:
        raise RuntimeError("Filing files were present but contained no usable text.")

    embeddings = make_embeddings(texts)
    out = pd.DataFrame(records)
    for i in range(NLP_EMBEDDING_DIMS):
        out[f"filing_embed_{i:02d}"] = embeddings[:, i]

    out.sort_values(["cik", "filing_date"], inplace=True)
    sims = []
    for _, g in out.groupby("cik", sort=False):
        emb = g[[f"filing_embed_{i:02d}" for i in range(NLP_EMBEDDING_DIMS)]].to_numpy()
        prev = np.vstack([np.full((1, emb.shape[1]), np.nan), emb[:-1]])
        dot = np.sum(emb * prev, axis=1)
        denom = np.linalg.norm(emb, axis=1) * np.linalg.norm(prev, axis=1)
        sims.extend(np.divide(dot, denom, out=np.full(len(g), np.nan), where=denom > 0))
    out["filing_similarity_prev"] = sims

    FILING_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FILING_FEATURES_OUT)
    print(f"Saved {FILING_FEATURES_OUT} ({len(out):,} filings, {out.shape[1]:,} columns)")


if __name__ == "__main__":
    main()

"""Dataset loading, validation and cleaning for the Enron spam dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import config

logger = logging.getLogger(__name__)

TEXT, LABEL = "text", "label"


@dataclass
class CleaningReport:
    rows_loaded: int = 0
    dropped_missing: int = 0
    dropped_empty_text: int = 0
    dropped_conflicting_labels: int = 0
    dropped_duplicates: int = 0
    rows_final: int = 0


def normalize_label(value) -> Optional[int]:
    """Map a raw label to 0 (ham) / 1 (spam); return None if unrecognised."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"spam", "1", "1.0", "true"}:
        return 1
    if s in {"ham", "legitimate", "0", "0.0", "false", "not spam"}:
        return 0
    return None


def _find_column(columns, candidates, explicit=None, required=True):
    lookup = {c.lower().strip(): c for c in columns}
    if explicit:
        if explicit.lower() not in lookup:
            raise ValueError(f"Column '{explicit}' not found. Available: {list(columns)}")
        return lookup[explicit.lower()]
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    if required:
        raise ValueError(
            f"Could not auto-detect a column from {candidates}. "
            f"Available columns: {list(columns)}. Pass the column name explicitly."
        )
    return None


def load_csv(path: Union[str, Path], text_col: Optional[str] = None,
             label_col: Optional[str] = None, subject_col: Optional[str] = None,
             encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV; join subject + body when a subject column exists."""
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    t = _find_column(df.columns, config.TEXT_COLUMN_CANDIDATES, text_col)
    l = _find_column(df.columns, config.LABEL_COLUMN_CANDIDATES, label_col)
    s = _find_column(df.columns, config.SUBJECT_COLUMN_CANDIDATES, subject_col, required=False)
    logger.info("Using columns -> text: %s | subject: %s | label: %s", t, s, l)

    body = df[t].fillna("").astype(str)
    if s is not None and s != t:
        text = df[s].fillna("").astype(str) + "\n" + body
    else:
        text = body
    return pd.DataFrame({TEXT: text.str.strip(), LABEL: df[l].map(normalize_label)})


def load_directory(root: Union[str, Path]) -> pd.DataFrame:
    """Load Enron-style folders: any directory named 'ham' or 'spam' holds .txt emails."""
    root = Path(root)
    records = []
    for folder_name, label in (("ham", 0), ("spam", 1)):
        for folder in root.rglob(folder_name):
            if not folder.is_dir():
                continue
            for f in folder.glob("*.txt"):
                records.append({TEXT: f.read_text(encoding="latin-1", errors="replace").strip(),
                                LABEL: label})
    if not records:
        raise FileNotFoundError(f"No ham/ or spam/ folders with .txt files found under {root}")
    return pd.DataFrame.from_records(records)


def load_raw(source: Optional[Union[str, Path]] = None, **csv_kwargs) -> pd.DataFrame:
    """Load from a CSV file, a directory (CSV or ham/spam folders). Defaults to data/."""
    source = Path(source) if source else config.DATA_DIR
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{source}'. Place the Enron CSV or extracted "
            f"enron1-6 folders inside '{config.DATA_DIR}' (see README)."
        )
    if source.is_file():
        return load_csv(source, **csv_kwargs)
    csvs = sorted(source.glob("*.csv"))
    if csvs:
        logger.info("Found CSV: %s", csvs[0])
        return load_csv(csvs[0], **csv_kwargs)
    return load_directory(source)


def clean_dataset(df: pd.DataFrame):
    """Drop missing, empty, conflicting and duplicate rows. Returns (df, CleaningReport)."""
    rep = CleaningReport(rows_loaded=len(df))

    n = len(df)
    df = df.dropna(subset=[TEXT, LABEL]).copy()
    rep.dropped_missing = n - len(df)

    n = len(df)
    df = df[df[TEXT].str.strip().str.len() > 0]
    rep.dropped_empty_text = n - len(df)

    # Same text with both labels = label noise -> remove all copies
    n = len(df)
    nunique = df.groupby(TEXT)[LABEL].transform("nunique")
    df = df[nunique == 1]
    rep.dropped_conflicting_labels = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset=[TEXT])
    rep.dropped_duplicates = n - len(df)

    df[LABEL] = df[LABEL].astype(int)
    df = df.reset_index(drop=True)
    rep.rows_final = len(df)
    return df, rep


def dataset_summary(df: pd.DataFrame) -> dict:
    counts = df[LABEL].value_counts().sort_index()
    lengths = df[TEXT].str.len()
    return {
        "n_emails": int(len(df)),
        "ham": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
        "spam_ratio": round(float(counts.get(1, 0)) / max(len(df), 1), 4),
        "chars_mean": round(float(lengths.mean()), 1),
        "chars_median": float(lengths.median()),
        "chars_max": int(lengths.max()),
    }


def load_dataset(source=None, **csv_kwargs):
    """Full pipeline: load -> clean -> summarise."""
    df, report = clean_dataset(load_raw(source, **csv_kwargs))
    return df, report, dataset_summary(df)


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)
    df, report, summary = load_dataset(sys.argv[1] if len(sys.argv) > 1 else None)
    print("Cleaning report:", json.dumps(asdict(report), indent=2))
    print("Summary:", json.dumps(summary, indent=2))
    print(df.head())
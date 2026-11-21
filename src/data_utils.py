"""Reusable helpers for loading and summarising the Diabetes 130-US Hospitals dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
# Codes that mean "unknown" in IDS_mapping.csv (NULL, Not Available, Not Mapped, Unknown/Invalid)
UNKNOWN_ID_CODES = {
    "admission_type_id": [5, 6, 8],
    "discharge_disposition_id": [18, 25, 26],
    "admission_source_id": [9, 15, 17, 20, 21],
}
# Discharge codes for patients who died or went to hospice: they cannot be readmitted
EXPIRED_OR_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]


def load_raw(path=RAW_DIR / "diabetic_data.csv") -> pd.DataFrame:
    """Read the raw file. Only '?' means missing: 'None' in max_glu_serum/A1Cresult means 'not tested'."""
    return pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)


def load_id_mapping(path=RAW_DIR / "IDS_mapping.csv") -> dict:
    """IDS_mapping.csv stacks 3 small tables separated by empty lines. Return {id column: Series code -> label}."""
    mappings, current, rows = {}, None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        code, _, label = line.partition(",")
        if code.strip() == "":                      # separator line between tables
            continue
        if code.endswith("_id"):                    # header of a new table
            if current:
                mappings[current] = pd.Series(dict(rows))
            current, rows = code, []
        else:
            rows.append((int(code), label.strip().strip('"')))
    mappings[current] = pd.Series(dict(rows))
    return mappings


def overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, number of distinct values, missing values, example values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "examples": [df[c].dropna().unique()[:4].tolist() for c in df.columns],
    })


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, counting '?' and the 'unknown' codes of the id columns."""
    flags = df.isna()
    for col, codes in UNKNOWN_ID_CODES.items():
        flags[col] = df[col].isin(codes)
    flags["gender"] = df["gender"].eq("Unknown/Invalid")
    table = pd.DataFrame({"n_missing": flags.sum(), "pct_missing": (flags.mean() * 100).round(2)})
    return table[table["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def numeric_summary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """describe() plus skewness and outlier counts by the IQR rule and by |z| > 3."""
    rows = {}
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std()
        rows[c] = {
            "mean": s.mean(), "median": s.median(), "std": s.std(), "min": s.min(), "max": s.max(),
            "skew": s.skew(),
            "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "z_outliers": int((z.abs() > 3).sum()),
        }
    return pd.DataFrame(rows).T.round(2)


def categorical_summary(df: pd.DataFrame, cols: list, rare_threshold=0.01) -> pd.DataFrame:
    """Distinct values, most frequent value and its share, and number of rare categories."""
    rows = {}
    for c in cols:
        freq = df[c].value_counts(normalize=True, dropna=True)
        rows[c] = {
            "n_unique": df[c].nunique(),
            "top": freq.index[0],
            "top_pct": round(freq.iloc[0] * 100, 2),
            "n_rare": int((freq < rare_threshold).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def icd9_group(code) -> str:
    """Group an ICD-9 code into the disease groups used by Strack et al. (2014)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code[0] in "EV":
        return "Other"
    value = float(code)
    if 390 <= value <= 459 or int(value) == 785:
        return "Circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "Respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "Digestive"
    if int(value) == 250:
        return "Diabetes"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def readmission_rate(df: pd.DataFrame, by: str, target="readmit_30") -> pd.DataFrame:
    """Share of encounters readmitted within 30 days per category of `by`, with the group size."""
    return (df.groupby(by, observed=True)[target]
              .agg(n="size", rate="mean")
              .assign(rate=lambda t: (t["rate"] * 100).round(2)))

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "publisher_id",
    "instrument_id",
    "symbol"
]

BAR_KEY = ["ts_event", "publisher_id", "instrument_id", "symbol"]
PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "publisher_id", "instrument_id"]

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()

def year_from_filename(path: Path) -> int:
    match = re.search(r"(\d{4})", path.name)
    if not match:
        raise ValueError(f"Could not extract year from filename: {path}")
    return int(match.group(1))

def _add_error(errors: pd.Series, mask: pd.Series, code: str) -> None:
    """Append a validation code to rows selected by mask."""
    errors.loc[mask] = errors.loc[mask].map(
        lambda current: f"{current};{code}" if current else code
    )

def read_source_file(path: Path, batch_id: str, ingestion_timestamp: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read one CSV and return raw Bronze rows, parsed working rows and a manifest."""

    # Read as strings so Bronze retains the source representation (including blanks).
    raw = pd.read_csv(path, dtype=str)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    # Keep only contract columns in the curated dataset, while retaining any extra
    # source columns in Bronze for traceability.
    raw.insert(0, "source_row_number", range(2, len(raw) + 2))  # CSV header is row 1
    raw.insert(0, "source_file", path.name)
    raw.insert(0, "batch_id", batch_id)
    raw.insert(0, "ingestion_timestamp_utc", ingestion_timestamp)
    raw.insert(0, "record_id", [
        hashlib.sha256(f"{path.name}:{row_number}".encode("utf-8")).hexdigest()
        for row_number in raw["source_row_number"]
    ])

    work = raw[["record_id", "source_file", "source_row_number", *REQUIRED_COLUMNS]].copy()
    work["source_year"] = year_from_filename(path)
    work["ingestion_timestamp_utc"] = ingestion_timestamp

    # Normalize whitespace before parsing. Original unmodified source values remain in Bronze.
    for col in REQUIRED_COLUMNS:
        work[col] = work[col].astype(str).str.strip()

    work["ts_event"] = pd.to_datetime(work["ts_event"], errors="coerce", utc=True)
    for col in NUMERIC_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Identifiers are integers in this dataset's schema. Invalid/non-integral values
    # are caught and quarantined instead of rounded.
    for col in ["publisher_id", "instrument_id", "volume"]:
        non_int = work[col].notna() & (work[col] % 1 != 0)
        work.loc[non_int, col] = pd.NA

    manifest = {
        "source_file": path.name,
        "source_file_sha256": sha256_file(path),
        "source_year_from_filename": year_from_filename(path),
        "source_row_count": int(len(work)),
        "source_columns": list(pd.read_csv(path, nrows=0).columns)        
    }

    return raw, work, manifest

def validate_and_deduplicate(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate records and split them into Silver and quarantine datasets."""
    frame = work.copy()
    errors = pd.Series("", index=frame.index, dtype="string")

    # Required fields.
    for column in REQUIRED_COLUMNS:
        missing = frame[column].isna() | (frame[column].astype("string").str.strip() == "")
        _add_error(errors, missing, f"MISSING_OR_INVALID_{column.upper()}")

    # Numeric and domain checks.
    for column in NUMERIC_COLUMNS:
        _add_error(errors, frame[column].isna(), f"INVALID_NUMERIC_{column.upper()}")

    bad_ohlc = (
        frame[["open", "high", "low", "close"]].notna().all(axis=1)
        & (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame["high"] < frame["low"])
        )
    )
    _add_error(errors, bad_ohlc, "INVALID_OHLC_RELATIONSHIP")
    _add_error(errors, frame["volume"].notna() & (frame["volume"] < 0), "NEGATIVE_VOLUME")
    _add_error(
        errors,
        frame["publisher_id"].notna() & (frame["publisher_id"] < 0),
        "NEGATIVE_PUBLISHER_ID",
    )
    _add_error(
        errors,
        frame["instrument_id"].notna() & (frame["instrument_id"] < 0),
        "NEGATIVE_INSTRUMENT_ID",
    )

    frame["validation_errors"] = errors
    initially_invalid = frame[frame["validation_errors"] != ""].copy()
    valid = frame[frame["validation_errors"] == ""].copy()

    # Duplicate key = same event timestamp + publisher + instrument + symbol.
    # Exact repeat copies: keep the first in Silver and quarantine extra copies.
    # Conflicting values for the same key: quarantine every version for investigation.
    if not valid.empty:
        duplicate_key = valid.duplicated(BAR_KEY, keep=False)
        fingerprint = valid[PRICE_VOLUME_COLUMNS].astype("string").fillna("<NULL>").agg(
            "|".join, axis=1
        )
        distinct_versions = fingerprint.groupby(
            [valid[column] for column in BAR_KEY], dropna=False
        ).transform("nunique")
        conflicting = duplicate_key & distinct_versions.gt(1)
        exact_extra = valid.duplicated(BAR_KEY + PRICE_VOLUME_COLUMNS, keep="first")
        exact_extra &= ~conflicting

        conflict_rows = valid[conflicting].copy()
        conflict_rows["validation_errors"] = "CONFLICTING_DUPLICATE_BAR_KEY"
        exact_duplicate_rows = valid[exact_extra].copy()
        exact_duplicate_rows["validation_errors"] = "EXACT_DUPLICATE_COPY"

        quarantine = pd.concat(
            [initially_invalid, conflict_rows, exact_duplicate_rows], ignore_index=True
        )
        silver = valid[~conflicting & ~exact_extra].copy()
    else:
        quarantine = initially_invalid.copy()
        silver = valid.copy()

    silver = silver.sort_values(["ts_event", "instrument_id"], kind="stable").reset_index(drop=True)
    quarantine = quarantine.reset_index(drop=True)

    # Store identifiers and volume as nullable integer types.
    for column in ["publisher_id", "instrument_id", "volume"]:
        if column in silver:
            silver[column] = silver[column].astype("Int64")
        if column in quarantine:
            quarantine[column] = quarantine[column].astype("Int64")

    stats = {
        "input_rows": int(len(frame)),
        "silver_rows": int(len(silver)),
        "quarantined_rows": int(len(quarantine)),
        "invalid_rows": int(len(initially_invalid)),
        "duplicate_or_conflict_rows": int(len(quarantine) - len(initially_invalid)),
        "distinct_event_timestamps": int(silver["ts_event"].nunique()) if not silver.empty else 0,
    }
    return silver, quarantine, stats


def run_ingestion(input_dir: Path, output_dir: Path, pattern: str = "*.csv") -> dict:
    """Ingest all matching CSVs in deterministic filename order."""
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files matching {pattern!r} found in {input_dir}")

    batch_id = str(uuid.uuid4())
    ingestion_timestamp = pd.Timestamp(datetime.now(timezone.utc))
    bronze_dir = output_dir / "bronze" / f"batch_id={batch_id}"
    silver_dir = output_dir / "silver" / f"batch_id={batch_id}"
    quarantine_dir = output_dir / "quarantine" / f"batch_id={batch_id}"
    for directory in [bronze_dir, silver_dir, quarantine_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    parsed_frames: list[pd.DataFrame] = []
    manifests: list[dict] = []
    bronze_counts: list[dict] = []

    for path in files:
        raw, work, manifest = read_source_file(path, batch_id, ingestion_timestamp)
        raw.to_parquet(bronze_dir / f"{path.stem}.parquet", index=False)
        parsed_frames.append(work)
        manifests.append(manifest)
        bronze_counts.append({"source_file": path.name, "bronze_rows": int(len(raw))})

    combined = pd.concat(parsed_frames, ignore_index=True)
    silver, quarantine, stats = validate_and_deduplicate(combined)

    silver_path = silver_dir / "gc_ohlcv_1m.parquet"
    silver.to_parquet(silver_path, index=False)
    quarantine_path = None
    if not quarantine.empty:
        quarantine_path = quarantine_dir / "gc_ohlcv_quarantine.parquet"
        quarantine.to_parquet(quarantine_path, index=False)

    summary = {
        "project": "GC Quant Data Platform",
        "batch_id": batch_id,
        "ingestion_timestamp_utc": ingestion_timestamp.isoformat(),
        "input_directory": str(input_dir),
        "file_count": len(files),
        "files": manifests,
        "bronze": bronze_counts,
        "quality": stats,
        "outputs": {
            "bronze_directory": str(bronze_dir),
            "silver_file": str(silver_path),
            "quarantine_file": str(quarantine_path) if quarantine_path else None,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"ingestion_summary_{batch_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest yearly GC OHLCV CSV files.")
    parser.add_argument("--input", type=Path, default=Path("data/input"), help="Directory of source CSVs")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--pattern", default="*.csv", help="Input filename glob (default: *.csv)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_ingestion(args.input, args.output, args.pattern)
    except Exception as exc:  # CLI boundary: display error and return non-zero.
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    quality = summary["quality"]
    print(f"Batch ID: {summary['batch_id']}")
    print(f"Files processed: {summary['file_count']}")
    print(f"Input rows: {quality['input_rows']:,}")
    print(f"Silver rows: {quality['silver_rows']:,}")
    print(f"Quarantined rows: {quality['quarantined_rows']:,}")
    print(f"Summary: {args.output / f'ingestion_summary_{summary['batch_id']}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
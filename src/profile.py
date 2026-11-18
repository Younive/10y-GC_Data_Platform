import argparse
import json
import sys
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
    "symbol",
]
BAR_KEY = ["ts_event", "publisher_id", "instrument_id", "symbol"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = [*PRICE_COLUMNS, "volume", "publisher_id", "instrument_id"]

GAP_BUCKETS = [
    (1, 5, ">1–5 minutes"),
    (5, 15, ">5–15 minutes"),
    (15, 60, ">15–60 minutes"),
    (60, 240, ">1–4 hours"),
    (240, 1440, ">4–24 hours"),
    (1440, float("inf"), ">24 hours"),
]


def _json_value(value):
    """Convert pandas/numpy scalars into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _record_dicts(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe row dictionaries."""
    return [
        {key: _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _gap_bucket(elapsed_minutes: float) -> str:
    for lower, upper, label in GAP_BUCKETS:
        if lower < elapsed_minutes <= upper:
            return label
    return "<=1 minute"


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def profile_frame(source: pd.DataFrame) -> dict[str, object]:
    """Create profiling summaries from an in-memory Silver OHLCV DataFrame.

    Returns a dictionary containing ``summary`` and report DataFrames. The input
    is copied and normalized; it is not modified in place.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"Silver dataset is missing required columns: {', '.join(missing_columns)}")
    if source.empty:
        raise ValueError("Silver dataset contains no rows")

    frame = source[REQUIRED_COLUMNS].copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    total_rows = int(len(frame))
    invalid_timestamp = frame["ts_event"].isna()
    valid_time = frame.loc[~invalid_timestamp].copy()
    if valid_time.empty:
        raise ValueError("Silver dataset has no parseable ts_event values")

    # Timestamp-level gaps are computed over distinct event timestamps, not rows.
    # This avoids creating artificial zero-length intervals when more than one
    # instrument appears at the same timestamp.
    unique_times = pd.Series(valid_time["ts_event"].drop_duplicates().sort_values().to_numpy())
    unique_times = pd.to_datetime(unique_times, utc=True)
    time_deltas = unique_times.diff()
    gap_mask = time_deltas > pd.Timedelta(minutes=1)
    gap_rows = pd.DataFrame(
        {
            "gap_start_utc": unique_times.shift(1),
            "gap_end_utc": unique_times,
            "elapsed_minutes": time_deltas.dt.total_seconds() / 60.0,
        }
    ).loc[gap_mask].copy()

    if not gap_rows.empty:
        gap_rows["year_utc"] = gap_rows["gap_end_utc"].dt.year.astype("int64")
        gap_rows["gap_bucket"] = gap_rows["elapsed_minutes"].map(_gap_bucket)
        # Number of missing minute-start timestamps strictly between the bars.
        gap_rows["estimated_missing_minute_starts"] = (
            gap_rows["elapsed_minutes"].floordiv(1).astype("int64") - 1
        ).clip(lower=0)
        gap_rows = gap_rows.sort_values("elapsed_minutes", ascending=False).reset_index(drop=True)
        largest_gaps = gap_rows.head(100).copy()
        gap_summary = (
            gap_rows.groupby(["year_utc", "gap_bucket"], as_index=False, observed=True)
            .agg(
                gap_count=("elapsed_minutes", "size"),
                total_elapsed_minutes=("elapsed_minutes", "sum"),
                estimated_missing_minute_starts=("estimated_missing_minute_starts", "sum"),
            )
            .sort_values(["year_utc", "gap_bucket"])
            .reset_index(drop=True)
        )
    else:
        gap_rows = _empty_frame(
            ["gap_start_utc", "gap_end_utc", "elapsed_minutes", "year_utc", "gap_bucket", "estimated_missing_minute_starts"]
        )
        largest_gaps = gap_rows.copy()
        gap_summary = _empty_frame(
            ["year_utc", "gap_bucket", "gap_count", "total_elapsed_minutes", "estimated_missing_minute_starts"]
        )

    valid_time["year_utc"] = valid_time["ts_event"].dt.year.astype("int64")
    valid_time["month_utc"] = valid_time["ts_event"].dt.strftime("%Y-%m")
    valid_time["utc_date"] = valid_time["ts_event"].dt.strftime("%Y-%m-%d")

    yearly = (
        valid_time.groupby("year_utc", as_index=False, observed=True)
        .agg(
            bars=("ts_event", "size"),
            first_bar_utc=("ts_event", "min"),
            last_bar_utc=("ts_event", "max"),
            utc_days=("utc_date", "nunique"),
            instruments=("instrument_id", "nunique"),
            low_min=("low", "min"),
            high_max=("high", "max"),
            volume_sum=("volume", "sum"),
            volume_mean=("volume", "mean"),
            volume_median=("volume", "median"),
            volume_p95=("volume", lambda values: values.quantile(0.95)),
            zero_volume_bars=("volume", lambda values: int((values == 0).sum())),
        )
        .sort_values("year_utc")
        .reset_index(drop=True)
    )

    monthly = (
        valid_time.groupby("month_utc", as_index=False, observed=True)
        .agg(
            bars=("ts_event", "size"),
            first_bar_utc=("ts_event", "min"),
            last_bar_utc=("ts_event", "max"),
            instruments=("instrument_id", "nunique"),
            low_min=("low", "min"),
            high_max=("high", "max"),
            volume_sum=("volume", "sum"),
            volume_mean=("volume", "mean"),
        )
        .sort_values("month_utc")
        .reset_index(drop=True)
    )

    daily = (
        valid_time.groupby("utc_date", as_index=False, observed=True)
        .agg(
            bars=("ts_event", "size"),
            first_bar_utc=("ts_event", "min"),
            last_bar_utc=("ts_event", "max"),
            instruments=("instrument_id", "nunique"),
            volume_sum=("volume", "sum"),
        )
        .sort_values("utc_date")
        .reset_index(drop=True)
    )

    instruments = (
        valid_time.groupby(["symbol", "instrument_id"], as_index=False, dropna=False, observed=True)
        .agg(
            bars=("ts_event", "size"),
            first_bar_utc=("ts_event", "min"),
            last_bar_utc=("ts_event", "max"),
            utc_days=("utc_date", "nunique"),
            low_min=("low", "min"),
            high_max=("high", "max"),
            volume_sum=("volume", "sum"),
            volume_mean=("volume", "mean"),
        )
        .sort_values(["first_bar_utc", "instrument_id"])
        .reset_index(drop=True)
    )

    # A timestamp with several instrument IDs is ambiguous for simple sequential
    # transition detection. Keep every timestamp in the sequence so an ambiguous
    # timestamp also breaks inference across that point instead of being skipped.
    by_timestamp = (
        valid_time.sort_values(["ts_event", "instrument_id"], kind="stable")
        .groupby("ts_event", as_index=False, observed=True)
        .agg(
            instrument_count=("instrument_id", "nunique"),
            instrument_id=("instrument_id", "first"),
            open=("open", "first"),
            close=("close", "last"),
        )
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    by_timestamp["previous_ts_event"] = by_timestamp["ts_event"].shift(1)
    by_timestamp["previous_instrument_id"] = by_timestamp["instrument_id"].shift(1)
    by_timestamp["previous_instrument_count"] = by_timestamp["instrument_count"].shift(1)
    by_timestamp["previous_close"] = by_timestamp["close"].shift(1)
    by_timestamp["previous_ts_gap_minutes"] = (
        by_timestamp["ts_event"] - by_timestamp["previous_ts_event"]
    ).dt.total_seconds() / 60.0
    transitions = by_timestamp[
        (by_timestamp["instrument_count"] == 1)
        & (by_timestamp["previous_instrument_count"] == 1)
        & by_timestamp["previous_instrument_id"].notna()
        & (by_timestamp["instrument_id"] != by_timestamp["previous_instrument_id"])
    ].copy()
    if not transitions.empty:
        transitions = transitions.rename(
            columns={
                "previous_ts_event": "previous_bar_utc",
                "ts_event": "current_bar_utc",
                "previous_instrument_id": "previous_instrument_id",
                "instrument_id": "current_instrument_id",
                "previous_close": "previous_close",
                "open": "current_open",
                "previous_ts_gap_minutes": "time_between_bars_minutes",
            }
        )
        transitions["open_minus_previous_close"] = (
            transitions["current_open"] - transitions["previous_close"]
        )
        transitions["open_gap_pct"] = (
            transitions["open_minus_previous_close"] / transitions["previous_close"] * 100.0
        ).where(transitions["previous_close"] != 0)
        transitions = transitions[
            [
                "previous_bar_utc",
                "current_bar_utc",
                "previous_instrument_id",
                "current_instrument_id",
                "previous_close",
                "current_open",
                "open_minus_previous_close",
                "open_gap_pct",
                "time_between_bars_minutes",
            ]
        ].reset_index(drop=True)
    else:
        transitions = _empty_frame(
            [
                "previous_bar_utc",
                "current_bar_utc",
                "previous_instrument_id",
                "current_instrument_id",
                "previous_close",
                "current_open",
                "open_minus_previous_close",
                "open_gap_pct",
                "time_between_bars_minutes",
            ]
        )

    price_frame = valid_time[PRICE_COLUMNS]
    bad_ohlc = (
        price_frame.notna().all(axis=1)
        & (
            (valid_time["high"] < valid_time[["open", "close", "low"]].max(axis=1))
            | (valid_time["low"] > valid_time[["open", "close", "high"]].min(axis=1))
            | (valid_time["high"] < valid_time["low"])
        )
    )
    non_positive_price = price_frame.notna().all(axis=1) & (price_frame <= 0).any(axis=1)
    missing_required = valid_time[REQUIRED_COLUMNS].isna().any(axis=1)
    for text_column in ["symbol"]:
        missing_required |= valid_time[text_column].astype("string").str.strip().eq("").fillna(False)
    negative_volume = valid_time["volume"].notna() & (valid_time["volume"] < 0)
    zero_volume = valid_time["volume"].notna() & (valid_time["volume"] == 0)
    minute_misaligned = (
        (valid_time["ts_event"].dt.second != 0)
        | (valid_time["ts_event"].dt.microsecond != 0)
        | (valid_time["ts_event"].dt.nanosecond != 0)
    )
    duplicate_bar_key = valid_time.duplicated(BAR_KEY, keep=False)
    duplicate_timestamp = valid_time.duplicated(["ts_event"], keep=False)

    checks = pd.DataFrame(
        [
            ("unparseable_timestamp_rows", int(invalid_timestamp.sum()), "Rows with ts_event that could not be parsed"),
            ("missing_required_field_rows", int(missing_required.sum()), "Rows with one or more null required fields"),
            ("invalid_ohlc_relationship_rows", int(bad_ohlc.sum()), "High/low do not contain open and close"),
            ("non_positive_price_rows", int(non_positive_price.sum()), "At least one OHLC value is zero or negative"),
            ("negative_volume_rows", int(negative_volume.sum()), "Volume is negative"),
            ("zero_volume_rows", int(zero_volume.sum()), "Volume equals zero; informational, not automatically invalid"),
            ("minute_misaligned_timestamp_rows", int(minute_misaligned.sum()), "Timestamp is not aligned to a whole-minute boundary"),
            ("duplicate_bar_key_rows", int(duplicate_bar_key.sum()), "Repeated timestamp/publisher/instrument/symbol key"),
            ("duplicate_timestamp_rows", int(duplicate_timestamp.sum()), "Repeated timestamp across rows; may include multiple instruments"),
        ],
        columns=["check_name", "row_count", "interpretation"],
    )
    checks["pct_of_rows"] = checks["row_count"] / total_rows * 100.0

    unique_symbol_values = sorted(valid_time["symbol"].dropna().astype(str).unique().tolist())
    unique_publisher_values = sorted(valid_time["publisher_id"].dropna().unique().tolist())
    unique_instrument_values = sorted(valid_time["instrument_id"].dropna().unique().tolist())
    first_timestamp = valid_time["ts_event"].min()
    last_timestamp = valid_time["ts_event"].max()
    overall_span_days = (last_timestamp - first_timestamp).total_seconds() / 86400.0

    summary = {
        "dataset": "GC continuous futures 1-minute OHLCV",
        "row_count": total_rows,
        "valid_timestamp_rows": int(len(valid_time)),
        "unparseable_timestamp_rows": int(invalid_timestamp.sum()),
        "first_event_timestamp_utc": first_timestamp.isoformat(),
        "last_event_timestamp_utc": last_timestamp.isoformat(),
        "calendar_span_days": round(overall_span_days, 3),
        "calendar_years_covered": sorted(valid_time["year_utc"].unique().astype(int).tolist()),
        "distinct_event_timestamps": int(valid_time["ts_event"].nunique()),
        "distinct_utc_dates": int(valid_time["utc_date"].nunique()),
        "symbols": unique_symbol_values,
        "publisher_ids": [_json_value(value) for value in unique_publisher_values],
        "instrument_ids": [_json_value(value) for value in unique_instrument_values],
        "instrument_transition_candidates": int(len(transitions)),
        "ambiguous_multi_instrument_timestamps": int((by_timestamp["instrument_count"] > 1).sum()),
        "timestamp_gaps_over_one_minute": int(len(gap_rows)),
        "estimated_missing_minute_starts_across_all_gaps": int(
            gap_rows["estimated_missing_minute_starts"].sum()
        ) if not gap_rows.empty else 0,
        "largest_gap_minutes": float(gap_rows["elapsed_minutes"].max()) if not gap_rows.empty else 0.0,
        "warning": (
            "Timestamp gaps are descriptive only. They can include exchange closures, "
            "holidays, no-trade intervals, and source-data gaps. Instrument transitions "
            "are candidates for contract rolls, not confirmed roll metadata."
        ),
    }

    return {
        "summary": summary,
        "yearly_coverage": yearly,
        "monthly_coverage": monthly,
        "daily_coverage": daily,
        "instrument_coverage": instruments,
        "gap_summary": gap_summary,
        "largest_gaps": largest_gaps,
        "instrument_transitions": transitions,
        "quality_checks": checks,
    }


def run_profile(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Read one Silver Parquet snapshot, write profiling CSVs and JSON summary."""
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input Parquet file does not exist: {input_path}")
    frame = pd.read_parquet(input_path, columns=REQUIRED_COLUMNS)
    results = profile_frame(frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = results["summary"]
    assert isinstance(summary, dict)
    for name, report in results.items():
        if name == "summary":
            continue
        assert isinstance(report, pd.DataFrame)
        report.to_csv(output_dir / f"{name}.csv", index=False)

    summary["input_file"] = str(input_path)
    summary_path = output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(summary, results, output_dir / "PROFILE_REPORT.md")
    return summary


def _write_markdown_report(summary: dict, results: dict, output_path: Path) -> None:
    checks = results["quality_checks"]
    yearly = results["yearly_coverage"]
    instruments = results["instrument_coverage"]
    transitions = results["instrument_transitions"]
    assert isinstance(checks, pd.DataFrame)
    assert isinstance(yearly, pd.DataFrame)
    assert isinstance(instruments, pd.DataFrame)
    assert isinstance(transitions, pd.DataFrame)

    lines = [
        "# GC Historical Data Profile",
        "",
        f"- Rows: **{summary['row_count']:,}**",
        f"- Time range (UTC): **{summary['first_event_timestamp_utc']}** to **{summary['last_event_timestamp_utc']}**",
        f"- Calendar span: **{summary['calendar_span_days']:,.3f} days**",
        f"- Symbols: **{', '.join(map(str, summary['symbols']))}**",
        f"- Distinct instrument IDs: **{len(summary['instrument_ids'])}**",
        f"- Candidate instrument transitions: **{summary['instrument_transition_candidates']}**",
        f"- Gaps over one minute: **{summary['timestamp_gaps_over_one_minute']:,}**",
        f"- Estimated missing minute starts across gaps: **{summary['estimated_missing_minute_starts_across_all_gaps']:,}**",
        "",
        "## Data-quality checks",
        "",
        "| Check | Rows | % of rows | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for row in checks.to_dict(orient="records"):
        lines.append(
            f"| {row['check_name']} | {int(row['row_count']):,} | {row['pct_of_rows']:.5f}% | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Yearly coverage",
            "",
            "| Year (UTC) | Bars | UTC dates | Instruments | Low | High | Volume sum |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in yearly.to_dict(orient="records"):
        lines.append(
            f"| {int(row['year_utc'])} | {int(row['bars']):,} | {int(row['utc_days']):,} | "
            f"{int(row['instruments']):,} | {row['low_min']} | {row['high_max']} | {row['volume_sum']} |"
        )
    lines.extend(
        [
            "",
            "## Instrument coverage",
            "",
            "| Symbol | Instrument ID | Bars | First UTC bar | Last UTC bar |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in instruments.to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {row['instrument_id']} | {int(row['bars']):,} | "
            f"{row['first_bar_utc']} | {row['last_bar_utc']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- A gap longer than one minute is not automatically an error. It can span scheduled exchange breaks, holidays, or minutes with no trades.",
            "- Instrument ID changes are **possible contract-roll candidates**, not definitive roll events. Confirm them with historical instrument-definition or symbology metadata.",
            "- Do not calculate strategy returns across instrument transitions until the continuous-contract roll methodology has been confirmed.",
            "- Coverage dates in this report are UTC calendar dates, not exchange trading dates.",
            "",
            summary["warning"],
            "",
            "See the companion CSV files for detailed yearly, monthly, daily, gap, instrument, and quality-check outputs.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a validated GC Silver Parquet dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one gc_ohlcv_1m.parquet Silver snapshot (not the parent folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/milestone2"),
        help="Directory where profile reports will be written.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_profile(args.input, args.output)
    except Exception as exc:  # CLI boundary: display a useful error and return non-zero.
        print(f"Profiling failed: {exc}", file=sys.stderr)
        return 1

    print(f"Rows profiled: {summary['row_count']:,}")
    print(f"Time range (UTC): {summary['first_event_timestamp_utc']} -> {summary['last_event_timestamp_utc']}")
    print(f"Instrument IDs: {len(summary['instrument_ids'])}")
    print(f"Possible instrument transitions: {summary['instrument_transition_candidates']}")
    print(f"Gaps over one minute: {summary['timestamp_gaps_over_one_minute']:,}")
    print(f"Profile report: {args.output / 'PROFILE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
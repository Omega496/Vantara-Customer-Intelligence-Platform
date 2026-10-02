"""End-to-end data pipeline script to load, clean, validate, and persist interim transactions."""

import json
import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.data.cleaning import clean_transactions
from src.data.loader import load_raw_excel
from src.data.validation import validate_cleaned_transactions
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_data_pipeline(
    raw_excel_path: Optional[str] = None,
    output_parquet_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, any]:
    """Runs the complete ingestion, cleaning, and validation data pipeline.

    Args:
        raw_excel_path: Optional override for raw excel path.
        output_parquet_path: Optional override for interim cleaned parquet output path.
        use_cache: Whether to use cached raw parquet if available.

    Returns:
        Audit and validation summary dictionary.
    """
    logger.info("==================================================")
    logger.info("Starting Milestone 1: Data Pipeline Execution")
    logger.info("==================================================")

    # 1. Ingestion
    raw_df = load_raw_excel(excel_path=raw_excel_path, use_cache=use_cache)
    logger.info(f"Loaded raw dataset with {len(raw_df):,} rows.")

    # 2. Cleaning & Wrangling
    cleaned_df, audit_stats = clean_transactions(
        df=raw_df,
        drop_missing_customers=True,
        remove_admin_codes=True,
        cap_outliers=True,
    )

    # 3. Validation
    validation_summary = validate_cleaned_transactions(cleaned_df)

    # 4. Persistence
    if output_parquet_path is None:
        output_parquet_path = ConfigManager.get(
            "paths.cleaned_transactions_file", "data/interim/cleaned_transactions.parquet"
        )

    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing cleaned transactions to {out_path}...")
    cleaned_df.to_parquet(out_path, index=False)

    summary_file = out_path.parent / "data_pipeline_summary.json"
    full_summary = {
        "audit_stats": audit_stats,
        "validation_summary": validation_summary,
        "output_file": str(out_path),
        "output_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, default=str)

    logger.info(f"Saved pipeline summary to {summary_file}")
    logger.info("Milestone 1 Data Pipeline finished successfully!")
    logger.info("==================================================")

    return full_summary


if __name__ == "__main__":
    run_data_pipeline()

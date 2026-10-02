"""Data ingestion and loading module for Online Retail II dataset."""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.data.validation import validate_raw_schema
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_excel(
    excel_path: Optional[str] = None,
    sheets: Optional[List[str]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Loads and combines multiple sheets of raw Online Retail II dataset.

    Args:
        excel_path: Path to the online_retail_II.xlsx file.
        sheets: Names of sheets to load. Defaults to ['Year 2009-2010', 'Year 2010-2011'].
        use_cache: If True, uses cached raw parquet file if present to speed up ingestion.

    Returns:
        Combined pandas DataFrame with standardized column names and sorted chronologically.

    Raises:
        FileNotFoundError: If the raw excel file does not exist.
    """
    if sheets is None:
        sheets = ConfigManager.get(
            "data_pipeline.sheets", ["Year 2009-2010", "Year 2010-2011"]
        )

    if excel_path is None:
        raw_dir = ConfigManager.get("paths.raw_data_dir", "data/raw")
        filename = ConfigManager.get("paths.raw_excel_filename", "online_retail_II.xlsx")
        excel_path = os.path.join(raw_dir, filename)

    excel_file = Path(excel_path)
    cache_path = excel_file.parent / "raw_transactions_cache.parquet"

    if use_cache and cache_path.exists():
        logger.info(f"Loading raw transactions from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df):,} records from cache.")
        return df

    if not excel_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {excel_file}. Please ensure file is downloaded."
        )

    logger.info(f"Loading sheets {sheets} from {excel_file}...")
    dfs: List[pd.DataFrame] = []
    for sheet in sheets:
        logger.info(f"Reading sheet: {sheet}")
        sheet_df = pd.read_excel(excel_file, sheet_name=sheet)
        validate_raw_schema(sheet_df)
        dfs.append(sheet_df)
        logger.info(f"Sheet '{sheet}' loaded with {len(sheet_df):,} rows.")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total raw rows combined: {len(combined_df):,}")

    # Standardize column names
    rename_map = ConfigManager.get(
        "data_pipeline.column_rename_map",
        {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        },
    )
    combined_df = combined_df.rename(columns=rename_map)

    # Standardize types
    combined_df["invoice"] = combined_df["invoice"].astype(str).str.strip()
    combined_df["stock_code"] = combined_df["stock_code"].astype(str).str.strip()
    combined_df["description"] = (
        combined_df["description"].fillna("").astype(str).str.strip()
    )
    combined_df["quantity"] = pd.to_numeric(combined_df["quantity"], errors="coerce")
    combined_df["price"] = pd.to_numeric(combined_df["price"], errors="coerce")
    combined_df["invoice_date"] = pd.to_datetime(
        combined_df["invoice_date"], errors="coerce"
    )
    combined_df["country"] = (
        combined_df["country"].fillna("Unspecified").astype(str).str.strip()
    )

    # Sort chronologically
    combined_df = combined_df.sort_values(by="invoice_date").reset_index(drop=True)

    if use_cache:
        logger.info(f"Saving raw combined cache to {cache_path}")
        combined_df.to_parquet(cache_path, index=False)

    return combined_df

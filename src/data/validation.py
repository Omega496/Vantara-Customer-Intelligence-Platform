"""Data validation module for transaction datasets."""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation constraints are violated."""
    pass


def validate_raw_schema(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> bool:
    """Validates that raw transaction dataframe contains all expected columns.

    Args:
        df: Input DataFrame to validate.
        expected_columns: List of required column names.

    Returns:
        True if validation passes.

    Raises:
        DataValidationError: If any required column is missing.
    """
    if expected_columns is None:
        expected_columns = [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        error_msg = f"Raw schema validation failed. Missing columns: {missing}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)

    logger.info("Raw schema validation passed.")
    return True


def validate_cleaned_transactions(
    df: pd.DataFrame,
    max_null_rate_customer_id: float = 0.0,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    require_positive_price: bool = True,
) -> Dict[str, any]:
    """Validates cleaned transactions data against business rules and data constraints.

    Args:
        df: Cleaned transactions DataFrame.
        max_null_rate_customer_id: Maximum allowable fraction of null customer IDs.
        min_date: Earliest expected invoice date.
        max_date: Latest expected invoice date.
        require_positive_price: Whether all prices must be strictly greater than 0.

    Returns:
        Dictionary of validation summary metrics.

    Raises:
        DataValidationError: If any critical validation check fails.
    """
    if df.empty:
        raise DataValidationError("Cleaned dataframe is empty.")

    required_cols = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
        "total_amount",
        "is_return",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns in cleaned data: {missing}")

    # Check null customer_id rate
    null_customer_rate = df["customer_id"].isnull().mean()
    if null_customer_rate > max_null_rate_customer_id:
        raise DataValidationError(
            f"Customer ID null rate {null_customer_rate:.4f} exceeds threshold {max_null_rate_customer_id}"
        )

    # Check price constraint
    if require_positive_price:
        invalid_prices = (df["price"] <= 0).sum()
        if invalid_prices > 0:
            raise DataValidationError(
                f"Found {invalid_prices} records with price <= 0 in cleaned data."
            )

    # Check quantity constraint (no zero quantity)
    zero_qty = (df["quantity"] == 0).sum()
    if zero_qty > 0:
        raise DataValidationError(f"Found {zero_qty} records with quantity == 0.")

    # Date bounds validation
    if min_date is None:
        min_date = datetime(2009, 11, 1)
    if max_date is None:
        max_date = datetime(2011, 12, 31)

    min_found = df["invoice_date"].min()
    max_found = df["invoice_date"].max()

    if min_found < min_date or max_found > max_date:
        logger.warning(
            f"Invoice date span [{min_found} to {max_found}] outside recommended span [{min_date} to {max_date}]."
        )

    summary = {
        "total_records": len(df),
        "unique_customers": df["customer_id"].nunique(),
        "unique_invoices": df["invoice"].nunique(),
        "date_range": (str(min_found), str(max_found)),
        "return_transactions": int(df["is_return"].sum()),
        "return_rate": float(df["is_return"].mean()),
        "total_revenue": float(df["total_amount"].sum()),
    }

    logger.info(f"Data validation passed successfully: {summary}")
    return summary

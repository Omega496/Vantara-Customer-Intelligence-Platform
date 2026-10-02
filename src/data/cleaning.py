"""Data cleaning and transaction wrangling module for Online Retail II."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes product descriptions by assigning the most frequent clean description per stock code.

    Args:
        df: Input DataFrame containing 'stock_code' and 'description'.

    Returns:
        DataFrame with standardized 'description' column.
    """
    df = df.copy()
    cleaned_desc = df["description"].astype(str).str.strip().str.upper()

    # Filter out empty or placeholder descriptions for lookup mapping
    valid_desc = df[cleaned_desc.str.len() > 2].copy()
    valid_desc["clean_desc"] = cleaned_desc[cleaned_desc.str.len() > 2]

    # Find mode description per stock_code
    mode_desc = (
        valid_desc.groupby(["stock_code", "clean_desc"])
        .size()
        .reset_index(name="count")
        .sort_values(["stock_code", "count"], ascending=[True, False])
        .drop_duplicates(subset=["stock_code"])
        .set_index("stock_code")["clean_desc"]
        .to_dict()
    )

    # Map most common clean description, fallback to cleaned original
    df["description"] = df["stock_code"].map(mode_desc).fillna(cleaned_desc)
    return df


def filter_administrative_codes(
    df: pd.DataFrame, admin_codes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int]:
    """Filters out non-product administrative stock codes (postage, bank charges, test codes).

    Args:
        df: Input DataFrame with 'stock_code'.
        admin_codes: List of administrative prefixes or codes to filter.

    Returns:
        Tuple of (filtered DataFrame, count of removed rows).
    """
    if admin_codes is None:
        admin_codes = ConfigManager.get(
            "data_pipeline.administrative_stock_codes",
            ["POST", "D", "M", "PADS", "DOT", "CRUK", "BANK CHARGES", "TEST", "AMAZONFEE"],
        )

    admin_set = {code.upper().strip() for code in admin_codes}
    code_upper = df["stock_code"].astype(str).str.upper().str.strip()

    mask = code_upper.isin(admin_set) | code_upper.str.startswith(("POST", "BANK", "CRUK"))
    removed_count = int(mask.sum())
    filtered_df = df[~mask].copy()

    logger.info(f"Filtered {removed_count:,} administrative records ({len(admin_set)} codes).")
    return filtered_df, removed_count


def handle_cancellations_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Explicitly flags cancellations and returns without discarding the behavioral return signal.

    Args:
        df: Input DataFrame with 'invoice' and 'quantity'.

    Returns:
        DataFrame with boolean 'is_return' column and standardized return quantities.
    """
    df = df.copy()
    invoice_str = df["invoice"].astype(str).str.strip()
    is_c_invoice = invoice_str.str.startswith("C", na=False)
    is_neg_qty = df["quantity"] < 0

    df["is_return"] = is_c_invoice | is_neg_qty

    # Ensure return quantities are strictly negative and normal purchases strictly positive
    # If an invoice had 'C' but positive quantity, correct quantity sign to negative
    df.loc[df["is_return"] & (df["quantity"] > 0), "quantity"] = -df.loc[
        df["is_return"] & (df["quantity"] > 0), "quantity"
    ]

    return df


def clean_transactions(
    df: pd.DataFrame,
    drop_missing_customers: bool = True,
    remove_admin_codes: bool = True,
    cap_outliers: bool = True,
    iqr_multiplier: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Executes full cleaning logic on transaction DataFrame.

    Steps:
    1. Filter missing customer_id (when drop_missing_customers=True)
    2. Convert customer_id to integer type
    3. Filter non-commercial / invalid prices (price <= 0)
    4. Flag returns / cancellations
    5. Filter administrative non-product codes
    6. Deduplicate identical transactions
    7. Standardize descriptions
    8. Calculate total_amount = quantity * price
    9. Detect / cap extreme data entry outliers

    Args:
        df: Raw combined DataFrame.
        drop_missing_customers: Whether to drop rows without Customer ID.
        remove_admin_codes: Whether to remove postage/administrative items.
        cap_outliers: Whether to cap extreme quantity/price values based on IQR bounds.
        iqr_multiplier: Multiplier for IQR bounds.

    Returns:
        Tuple of (Cleaned DataFrame, Audit statistics dictionary).
    """
    initial_rows = len(df)
    logger.info(f"Starting data cleaning on {initial_rows:,} raw transactions.")

    cleaned = df.copy()

    # 1. Missing customer ID handling
    missing_cust_count = int(cleaned["customer_id"].isnull().sum())
    if drop_missing_customers:
        cleaned = cleaned.dropna(subset=["customer_id"]).copy()
        cleaned["customer_id"] = cleaned["customer_id"].astype(int)
        logger.info(
            f"Dropped {missing_cust_count:,} records with missing customer_id. Remaining: {len(cleaned):,}"
        )

    # 2. Filter non-commercial / invalid prices
    min_unit_price = ConfigManager.get("data_pipeline.validation.min_unit_price", 0.001)
    invalid_price_mask = cleaned["price"].isnull() | (cleaned["price"] < min_unit_price)
    invalid_price_count = int(invalid_price_mask.sum())
    cleaned = cleaned[~invalid_price_mask].copy()
    logger.info(f"Removed {invalid_price_count:,} records with price < {min_unit_price}.")

    # 3. Filter zero quantities
    zero_qty_mask = cleaned["quantity"] == 0
    zero_qty_count = int(zero_qty_mask.sum())
    cleaned = cleaned[~zero_qty_mask].copy()

    # 4. Handle cancellations and returns
    cleaned = handle_cancellations_and_returns(cleaned)

    # 5. Filter administrative codes
    admin_removed = 0
    if remove_admin_codes:
        cleaned, admin_removed = filter_administrative_codes(cleaned)

    # 6. Deduplicate exact duplicate transaction records
    dedup_subset = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
    ]
    duplicate_count = int(cleaned.duplicated(subset=dedup_subset).sum())
    cleaned = cleaned.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    logger.info(f"Removed {duplicate_count:,} exact duplicate lines.")

    # 7. Standardize descriptions
    cleaned = standardize_descriptions(cleaned)

    # 8. Compute total amount
    cleaned["total_amount"] = (cleaned["quantity"] * cleaned["price"]).round(2)

    # 9. Outlier handling
    outlier_stats = {}
    if cap_outliers:
        pos_df = cleaned[~cleaned["is_return"]]

        # Quantity upper bound
        q1_q, q3_q = pos_df["quantity"].quantile(0.25), pos_df["quantity"].quantile(0.75)
        iqr_q = q3_q - q1_q
        qty_upper = q3_q + (iqr_multiplier * iqr_q)

        # Price upper bound
        q1_p, q3_p = pos_df["price"].quantile(0.25), pos_df["price"].quantile(0.75)
        iqr_p = q3_p - q1_p
        price_upper = q3_p + (iqr_multiplier * iqr_p)

        # Extreme data entry thresholds (sanity cap)
        hard_max_qty = max(qty_upper, 5000)
        hard_max_price = max(price_upper, 2000)

        extreme_qty_mask = cleaned["quantity"].abs() > hard_max_qty
        extreme_price_mask = cleaned["price"] > hard_max_price
        extreme_records = int((extreme_qty_mask | extreme_price_mask).sum())

        logger.info(
            f"Filtering extreme data entry errors beyond hard caps (Qty > {hard_max_qty}, Price > {hard_max_price}): {extreme_records} rows."
        )
        cleaned = cleaned[~(extreme_qty_mask | extreme_price_mask)].copy()

        outlier_stats = {
            "hard_max_qty": hard_max_qty,
            "hard_max_price": hard_max_price,
            "extreme_records_removed": extreme_records,
        }

    # Final chronological sort
    cleaned = cleaned.sort_values(by="invoice_date").reset_index(drop=True)

    audit_stats = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "missing_customers_dropped": missing_cust_count,
        "invalid_prices_dropped": invalid_price_count,
        "zero_quantities_dropped": zero_qty_count,
        "admin_codes_removed": admin_removed,
        "duplicates_removed": duplicate_count,
        "return_rows_count": int(cleaned["is_return"].sum()),
        "unique_customers": int(cleaned["customer_id"].nunique()),
        "outlier_stats": outlier_stats,
    }

    logger.info(f"Data cleaning complete. Final rows: {len(cleaned):,}")
    return cleaned, audit_stats

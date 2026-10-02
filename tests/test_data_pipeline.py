"""Unit tests for the data pipeline, cleaning, and validation modules."""

from datetime import datetime

import pandas as pd
import pytest

from src.data.cleaning import (
    clean_transactions,
    filter_administrative_codes,
    handle_cancellations_and_returns,
    standardize_descriptions,
)
from src.data.validation import (
    DataValidationError,
    validate_cleaned_transactions,
    validate_raw_schema,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing a sample transaction DataFrame for testing."""
    return pd.DataFrame(
        {
            "Invoice": ["489434", "C489435", "489436", "489437", "489438", "489439"],
            "StockCode": ["85048", "85048", "POST", "21232", "21232", "79323P"],
            "Description": [
                "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
                "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
                "POSTAGE",
                "STRAWBERRY CERAMIC TRINKET BOX",
                "strawberry ceramic trinket box",  # lowercase typo test
                "PINK CHERRY BLOSSOM CUP & SAUCER",
            ],
            "Quantity": [12, -1, 1, 24, 0, 8],
            "InvoiceDate": [
                datetime(2009, 12, 1, 7, 45),
                datetime(2009, 12, 1, 7, 46),
                datetime(2009, 12, 1, 8, 0),
                datetime(2009, 12, 1, 8, 30),
                datetime(2009, 12, 1, 9, 0),
                datetime(2009, 12, 1, 9, 15),
            ],
            "Price": [6.95, 6.95, 18.00, 1.25, 0.0, 2.95],
            "Customer ID": [13085.0, 13085.0, 13078.0, 15311.0, None, 14527.0],
            "Country": ["United Kingdom"] * 6,
        }
    )


def test_validate_raw_schema_success(sample_raw_data: pd.DataFrame):
    """Verifies that schema validation passes on compliant raw data."""
    assert validate_raw_schema(sample_raw_data) is True


def test_validate_raw_schema_failure():
    """Verifies that schema validation raises DataValidationError when columns are missing."""
    invalid_df = pd.DataFrame({"Invoice": ["1"], "Quantity": [10]})
    with pytest.raises(DataValidationError):
        validate_raw_schema(invalid_df)


def test_handle_cancellations_and_returns():
    """Verifies that cancellations and negative quantities are correctly flagged."""
    df = pd.DataFrame(
        {
            "invoice": ["489434", "C489435", "489436"],
            "quantity": [10, -2, -5],
        }
    )
    res = handle_cancellations_and_returns(df)
    assert "is_return" in res.columns
    assert list(res["is_return"]) == [False, True, True]
    assert list(res["quantity"]) == [10, -2, -5]


def test_filter_administrative_codes():
    """Verifies that non-product administrative items are filtered out."""
    df = pd.DataFrame(
        {
            "stock_code": ["85048", "POST", "BANK CHARGES", "21232"],
            "description": ["Item 1", "Postage", "Fee", "Item 2"],
        }
    )
    filtered, count = filter_administrative_codes(df)
    assert count == 2
    assert list(filtered["stock_code"]) == ["85048", "21232"]


def test_standardize_descriptions():
    """Verifies that product descriptions are standardized to uppercase mode description."""
    df = pd.DataFrame(
        {
            "stock_code": ["21232", "21232", "21232"],
            "description": [
                "STRAWBERRY CERAMIC TRINKET BOX",
                "strawberry ceramic trinket box",
                "STRAWBERRY CERAMIC TRINKET BOX",
            ],
        }
    )
    res = standardize_descriptions(df)
    assert (res["description"] == "STRAWBERRY CERAMIC TRINKET BOX").all()


def test_clean_transactions_end_to_end(sample_raw_data: pd.DataFrame):
    """Verifies complete cleaning function correctly applies all rules."""
    # Rename columns to match raw loader output
    raw = sample_raw_data.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )

    cleaned, audit = clean_transactions(
        raw, drop_missing_customers=True, remove_admin_codes=True
    )

    # Missing customer (row 4) was dropped
    assert cleaned["customer_id"].isnull().sum() == 0
    # Zero price (row 4) was dropped
    assert (cleaned["price"] <= 0).sum() == 0
    # Admin code POST (row 2) was dropped
    assert "POST" not in cleaned["stock_code"].values
    # Total amount computed
    assert "total_amount" in cleaned.columns
    # Return flag computed
    assert "is_return" in cleaned.columns
    assert audit["missing_customers_dropped"] >= 1


def test_validate_cleaned_transactions_success():
    """Verifies validation succeeds on properly formatted clean dataset."""
    valid_df = pd.DataFrame(
        {
            "invoice": ["489434", "C489435"],
            "stock_code": ["85048", "85048"],
            "description": ["GLASS BALL", "GLASS BALL"],
            "quantity": [10, -2],
            "invoice_date": [
                datetime(2010, 1, 15, 10, 0),
                datetime(2010, 1, 16, 11, 0),
            ],
            "price": [5.0, 5.0],
            "customer_id": [12345, 12345],
            "country": ["United Kingdom", "United Kingdom"],
            "total_amount": [50.0, -10.0],
            "is_return": [False, True],
        }
    )
    summary = validate_cleaned_transactions(valid_df)
    assert summary["total_records"] == 2
    assert summary["unique_customers"] == 1
    assert summary["return_transactions"] == 1


def test_validate_cleaned_transactions_failure_on_null_customer():
    """Verifies validation raises error if customer_id contains nulls."""
    invalid_df = pd.DataFrame(
        {
            "invoice": ["489434"],
            "stock_code": ["85048"],
            "description": ["GLASS BALL"],
            "quantity": [10],
            "invoice_date": [datetime(2010, 1, 15)],
            "price": [5.0],
            "customer_id": [None],
            "country": ["United Kingdom"],
            "total_amount": [50.0],
            "is_return": [False],
        }
    )
    with pytest.raises(DataValidationError):
        validate_cleaned_transactions(invalid_df, max_null_rate_customer_id=0.0)


def test_loader_cache_loading(tmp_path):
    """Verifies that load_raw_excel utilizes existing parquet cache correctly."""
    from src.data.loader import load_raw_excel

    sample = pd.DataFrame(
        {
            "invoice": ["1001"],
            "stock_code": ["TEST"],
            "description": ["SAMPLE"],
            "quantity": [1],
            "price": [10.0],
            "customer_id": [12345],
            "invoice_date": [datetime(2010, 1, 1)],
            "country": ["UK"],
        }
    )
    cache_file = tmp_path / "raw_transactions_cache.parquet"
    sample.to_parquet(cache_file)

    loaded = load_raw_excel(excel_path=str(tmp_path / "fake.xlsx"), use_cache=True)
    assert len(loaded) == 1
    assert loaded.iloc[0]["invoice"] == "1001"


def test_run_data_pipeline_end_to_end(tmp_path):
    """Verifies that run_data_pipeline processes raw transactions and generates output files."""
    from src.data.pipeline import run_data_pipeline

    sample = pd.DataFrame(
        {
            "invoice": ["1001", "1002"],
            "stock_code": ["85048", "21232"],
            "description": ["GLASS BALL", "TRINKET BOX"],
            "quantity": [10, 5],
            "price": [2.5, 4.0],
            "customer_id": [12345, 12346],
            "invoice_date": [datetime(2010, 1, 1), datetime(2010, 1, 2)],
            "country": ["United Kingdom", "France"],
        }
    )
    cache_file = tmp_path / "raw_transactions_cache.parquet"
    sample.to_parquet(cache_file)

    out_file = tmp_path / "cleaned.parquet"
    summary = run_data_pipeline(
        raw_excel_path=str(tmp_path / "dummy.xlsx"),
        output_parquet_path=str(out_file),
        use_cache=True,
    )

    assert out_file.exists()
    assert summary["validation_summary"]["total_records"] == 2
    assert summary["validation_summary"]["unique_customers"] == 2


def test_perform_eda(tmp_path):
    """Verifies that perform_eda generates distribution figures and summary findings."""
    from src.data.eda import perform_eda

    df = pd.DataFrame(
        {
            "invoice": ["1", "2", "3", "4"],
            "stock_code": ["A", "B", "A", "C"],
            "description": ["Item A", "Item B", "Item A", "Item C"],
            "quantity": [5, 10, -1, 4],
            "price": [10.0, 20.0, 10.0, 15.0],
            "customer_id": [101, 102, 101, 103],
            "invoice_date": [
                datetime(2010, 1, 15),
                datetime(2010, 2, 20),
                datetime(2010, 2, 21),
                datetime(2010, 3, 10),
            ],
            "country": ["United Kingdom", "Germany", "United Kingdom", "France"],
            "total_amount": [50.0, 200.0, -10.0, 60.0],
            "is_return": [False, False, True, False],
        }
    )
    findings = perform_eda(df, output_dir=str(tmp_path / "figures"))
    assert findings["total_active_customers"] == 3
    assert len(findings["hypotheses"]) == 4
    assert (tmp_path / "figures" / "eda_monthly_revenue_trend.png").exists()
    assert (tmp_path / "figures" / "eda_country_revenue.png").exists()
    assert (tmp_path / "figures" / "eda_rfm_distributions.png").exists()
    assert (tmp_path / "figures" / "eda_return_rate_distribution.png").exists()




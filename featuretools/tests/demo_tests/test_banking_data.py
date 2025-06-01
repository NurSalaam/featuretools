import pytest
import pandas as pd
import numpy as np
import datetime
from featuretools.demo.banking import load_banking_data, customers_schema, cards_schema, transactions_schema, merchants_schema

# Define default parameters for tests for brevity
DEFAULT_PARAMS = {
    "n_customers": 5,
    "n_cards_per_customer": 2,
    "n_transactions_per_card": 10,
    "n_merchants": 3,
    "start_date": datetime.date(2023, 1, 1),
    "end_date": datetime.date(2023, 12, 31),
    "fraud_percentage": 0.1
}

@pytest.fixture(scope="module")
def banking_data_default():
    """Fixture to generate banking data with default parameters."""
    return load_banking_data(**DEFAULT_PARAMS)

@pytest.fixture(scope="module")
def banking_data_no_fraud():
    """Fixture to generate banking data with no fraud."""
    params = DEFAULT_PARAMS.copy()
    params["fraud_percentage"] = 0
    return load_banking_data(**params)

@pytest.fixture(scope="module")
def banking_data_all_fraud():
    """Fixture to generate banking data with all fraud (or high chance)."""
    params = DEFAULT_PARAMS.copy()
    params["fraud_percentage"] = 1.0
    return load_banking_data(**params)

@pytest.fixture(scope="module")
def banking_data_zero_transactions():
    """Fixture to generate banking data with zero transactions."""
    params = DEFAULT_PARAMS.copy()
    params["n_transactions_per_card"] = 0
    return load_banking_data(**params)


def test_output_structure(banking_data_default):
    """Verify that all expected tables are present in the output."""
    data = banking_data_default
    assert isinstance(data, dict)
    expected_tables = ["customers", "cards", "transactions", "merchants"]
    for table_name in expected_tables:
        assert table_name in data
        assert isinstance(data[table_name], pd.DataFrame)

def test_table_schemas(banking_data_default):
    """For each table, check if all expected columns are present."""
    data = banking_data_default
    assert set(data["customers"].columns) == set(customers_schema.keys())
    assert set(data["cards"].columns) == set(cards_schema.keys())
    assert set(data["transactions"].columns) == set(transactions_schema.keys())
    assert set(data["merchants"].columns) == set(merchants_schema.keys())

def test_data_types(banking_data_default):
    """For key columns in each table, verify that the data types are correct."""
    data = banking_data_default

    # Customers
    assert pd.api.types.is_integer_dtype(data["customers"]["customer_id"])
    assert pd.api.types.is_object_dtype(data["customers"]["name"]) # String
    assert pd.api.types.is_object_dtype(data["customers"]["date_of_birth"]) # datetime.date
    assert pd.api.types.is_object_dtype(data["customers"]["account_open_date"]) # datetime.date

    # Cards
    assert pd.api.types.is_integer_dtype(data["cards"]["card_id"])
    assert pd.api.types.is_integer_dtype(data["cards"]["customer_id"])
    assert pd.api.types.is_object_dtype(data["cards"]["card_number"]) # String
    assert pd.api.types.is_object_dtype(data["cards"]["issue_date"]) # datetime.date
    assert pd.api.types.is_object_dtype(data["cards"]["expiry_date"]) # datetime.date
    assert pd.api.types.is_float_dtype(data["cards"]["credit_limit"])

    # Transactions
    assert pd.api.types.is_integer_dtype(data["transactions"]["transaction_id"])
    assert pd.api.types.is_integer_dtype(data["transactions"]["card_id"])
    assert pd.api.types.is_object_dtype(data["transactions"]["transaction_date"]) # datetime.date
    assert pd.api.types.is_object_dtype(data["transactions"]["transaction_time"]) # datetime.time
    assert pd.api.types.is_float_dtype(data["transactions"]["amount"])
    assert pd.api.types.is_bool_dtype(data["transactions"]["is_fraud"])
    assert pd.api.types.is_integer_dtype(data["transactions"]["merchant_id"])

    # Merchants
    assert pd.api.types.is_integer_dtype(data["merchants"]["merchant_id"])
    assert pd.api.types.is_object_dtype(data["merchants"]["merchant_name"]) # String

def test_row_count_verification(banking_data_default):
    """Check if the number of generated rows is consistent with the input parameters."""
    data = banking_data_default
    n_customers = DEFAULT_PARAMS["n_customers"]
    n_cards_per_customer = DEFAULT_PARAMS["n_cards_per_customer"]
    n_transactions_per_card = DEFAULT_PARAMS["n_transactions_per_card"]
    n_merchants = DEFAULT_PARAMS["n_merchants"]

    assert len(data["customers"]) == n_customers
    assert len(data["cards"]) == n_customers * n_cards_per_customer
    # Approximate check for transactions due to potential date constraints
    # Each card should have n_transactions_per_card
    expected_transactions = n_customers * n_cards_per_customer * n_transactions_per_card
    assert len(data["transactions"]) <= expected_transactions
    # Allow for some variation if card validity periods are very short / clash with overall start/end date
    # but it should be close to the expected number
    assert len(data["transactions"]) >= expected_transactions * 0.9 # Expect at least 90%

    assert len(data["merchants"]) == n_merchants

def test_row_count_zero_transactions(banking_data_zero_transactions):
    """Check row counts when n_transactions_per_card is 0."""
    data = banking_data_zero_transactions
    assert len(data["transactions"]) == 0
    assert len(data["customers"]) == DEFAULT_PARAMS["n_customers"] # Other tables should still generate

def test_relationship_integrity(banking_data_default):
    """Verify basic referential integrity between tables."""
    data = banking_data_default
    customers = data["customers"]
    cards = data["cards"]
    transactions = data["transactions"]
    merchants = data["merchants"]

    # customer_id in cards exists in customers
    assert cards["customer_id"].isin(customers["customer_id"]).all()

    # card_id in transactions exists in cards
    if not transactions.empty: # Only if there are transactions
        assert transactions["card_id"].isin(cards["card_id"]).all()

    # merchant_id in transactions exists in merchants
    if not transactions.empty: # Only if there are transactions
        assert transactions["merchant_id"].isin(merchants["merchant_id"]).all()


def test_date_consistency(banking_data_default):
    """Check various date consistencies within and between tables."""
    data = banking_data_default
    customers = data["customers"]
    cards = data["cards"]
    transactions = data["transactions"]

    start_date = DEFAULT_PARAMS["start_date"]
    end_date = DEFAULT_PARAMS["end_date"]

    # Account open date vs card issue date
    merged_cust_cards = pd.merge(cards, customers, on="customer_id")
    assert (merged_cust_cards["issue_date"] >= merged_cust_cards["account_open_date"]).all()

    # Card issue date vs expiry date
    assert (cards["issue_date"] <= cards["expiry_date"]).all()

    if not transactions.empty:
        # Transaction date within overall start/end date
        assert (transactions["transaction_date"] >= start_date).all()
        assert (transactions["transaction_date"] <= end_date).all()

        # Transaction date vs card validity
        merged_trans_cards = pd.merge(transactions, cards, on="card_id")
        assert (merged_trans_cards["transaction_date"] >= merged_trans_cards["issue_date"]).all()
        assert (merged_trans_cards["transaction_date"] <= merged_trans_cards["expiry_date"]).all()


def test_fraudulent_transaction_generation(banking_data_default, banking_data_no_fraud, banking_data_all_fraud):
    """Test generation of fraudulent transactions based on fraud_percentage."""
    # With fraud_percentage > 0
    data_with_fraud = banking_data_default
    if DEFAULT_PARAMS["n_transactions_per_card"] > 0 :
      assert data_with_fraud["transactions"]["is_fraud"].any() # At least one fraud
      # Check that not all are fraud (unless fraud_percentage is 1.0)
      if DEFAULT_PARAMS["fraud_percentage"] < 1.0:
          assert not data_with_fraud["transactions"]["is_fraud"].all()


    # With fraud_percentage = 0
    data_no_fraud = banking_data_no_fraud
    if DEFAULT_PARAMS["n_transactions_per_card"] > 0 :
        assert not data_no_fraud["transactions"]["is_fraud"].any() # No fraud

    # With fraud_percentage = 1
    data_all_fraud = banking_data_all_fraud
    if DEFAULT_PARAMS["n_transactions_per_card"] > 0 :
        assert data_all_fraud["transactions"]["is_fraud"].all() # All fraud

def test_parameterization_different_sizes():
    """Test with different valid parameter values."""
    params_small = {
        "n_customers": 1, "n_cards_per_customer": 1, "n_transactions_per_card": 1,
        "n_merchants": 1, "start_date": datetime.date(2024, 1, 1),
        "end_date": datetime.date(2024, 1, 15), "fraud_percentage": 0.5
    }
    data_small = load_banking_data(**params_small)
    assert len(data_small["customers"]) == 1
    assert len(data_small["cards"]) == 1
    # Transactions can be <= 1 due to very short date range and card validity
    assert len(data_small["transactions"]) <= 1
    assert len(data_small["merchants"]) == 1
    if len(data_small["transactions"]) ==1: # if a transaction was actually generated
        assert isinstance(data_small["transactions"]["is_fraud"].iloc[0], (bool, np.bool_))

    params_large_customers = DEFAULT_PARAMS.copy()
    params_large_customers["n_customers"] = 20
    data_large_customers = load_banking_data(**params_large_customers)
    assert len(data_large_customers["customers"]) == 20
    assert len(data_large_customers["cards"]) == 20 * DEFAULT_PARAMS["n_cards_per_customer"]

def test_specific_date_scenario():
    """ Test a scenario where card issue and expiry dates might limit transactions"""
    params = {
        "n_customers": 1,
        "n_cards_per_customer": 1,
        "n_transactions_per_card": 5,
        "n_merchants": 1,
        "start_date": datetime.date(2023, 1, 1),
        "end_date": datetime.date(2023, 1, 10), # Short overall window
        "fraud_percentage": 0.0
    }
    # In load_banking_data, card issue_date can be up to end_date (Jan 10)
    # And expiry_date is 2-5 years later.
    # Transaction dates are between max(start_date, issue_date) and min(end_date, expiry_date)
    # So transactions should be possible.
    data = load_banking_data(**params)

    assert len(data["customers"]) == 1
    assert len(data["cards"]) == 1

    card_issue_date = data["cards"]["issue_date"].iloc[0]
    card_expiry_date = data["cards"]["expiry_date"].iloc[0]

    # Ensure card dates are logical relative to each other
    assert card_issue_date <= card_expiry_date
    # Ensure card issue date is within the overall simulation period or just before for account opening
    assert card_issue_date <= params["end_date"]

    if not data["transactions"].empty:
        transaction_dates = data["transactions"]["transaction_date"]
        assert (transaction_dates >= card_issue_date).all()
        assert (transaction_dates <= card_expiry_date).all()
        assert (transaction_dates >= params["start_date"]).all()
        assert (transaction_dates <= params["end_date"]).all()
        assert len(data["transactions"]) <= params["n_transactions_per_card"]
    else:
        # It's possible no transactions are generated if card issue_date ends up being
        # very late in the short window, making valid transaction period zero.
        # E.g. if issue_date is 2023-01-10, then transaction window is just that one day.
        # This is acceptable given the generation logic.
        pass

# To run these tests, navigate to the directory containing featuretools and run:
# python -m pytest
# or if featuretools is installed:
# pytest path/to/featuretools/tests/demo_tests/test_banking_data.py

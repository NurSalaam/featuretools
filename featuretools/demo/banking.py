import pandas as pd
import numpy as np
import datetime
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define the schema for the banking data

# Customers table
customers_schema = {
    "customer_id": np.int64,
    "name": str,
    "email": str,
    "phone": str,
    "address": str,
    "city": str,
    "state": str,
    "zip_code": str,
    "date_of_birth": object, # Pandas will handle datetime.date
    "account_open_date": object # Pandas will handle datetime.date
}

# Cards table
cards_schema = {
    "card_id": np.int64,
    "customer_id": np.int64,
    "card_number": str,
    "card_type": str,  # e.g., debit, credit
    "card_provider": str,  # e.g., Visa, Mastercard
    "issue_date": object, # Pandas will handle datetime.date
    "expiry_date": object, # Pandas will handle datetime.date
    "cvv": str,
    "credit_limit": np.float64
}

# Transactions table
transactions_schema = {
    "transaction_id": np.int64,
    "card_id": np.int64,
    "transaction_date": object, # Pandas will handle datetime.date
    "transaction_time": object, # Pandas will handle datetime.time
    "amount": np.float64,
    "merchant_id": np.int64,
    "location": str,  # e.g., city, state, country
    "transaction_type": str,  # e.g., purchase, withdrawal, transfer
    "is_fraud": bool
}

# Merchants table
merchants_schema = {
    "merchant_id": np.int64,
    "merchant_name": str,
    "merchant_category": str,  # e.g., retail, dining, online
    "city": str,
    "state": str,
    "country": str
}

def load_banking_data(n_customers=100, n_cards_per_customer=2, n_transactions_per_card=100,
                        n_merchants=50, start_date=datetime.date(2023, 1, 1),
                        end_date=datetime.date(2023, 12, 31), fraud_percentage=0.05):
    """
    Generates a synthetic dataset for a banking scenario, including customers,
    credit/debit cards, transactions, and merchants.

    This function uses the Faker library to create realistic-looking data.
    It allows for customization of data size and fraud characteristics.

    Args:
        n_customers (int, optional): Number of unique customers to generate.
            Defaults to 100.
        n_cards_per_customer (int, optional): Number of cards (debit or credit)
            to generate for each customer. Defaults to 2.
        n_transactions_per_card (int, optional): Number of transactions to generate
            for each card. Note that the actual number might be slightly less if
            date constraints (card expiry, overall date range) limit possible
            transaction dates. Defaults to 100.
        n_merchants (int, optional): Number of unique merchants to generate.
            Defaults to 50.
        start_date (datetime.date, optional): The earliest possible date for
            account openings and transactions. Defaults to January 1, 2023.
        end_date (datetime.date, optional): The latest possible date for
            account openings and transactions. Defaults to December 31, 2023.
        fraud_percentage (float, optional): The approximate percentage of transactions
            that should be marked as fraudulent (value between 0.0 and 1.0).
            Defaults to 0.05 (5%).

    Returns:
        dict: A dictionary where keys are table names (strings) and values are
        pandas DataFrames. The tables included are:
            - "customers": Contains customer information.
                Key columns: customer_id (PK), name, email, phone, address,
                             city, state, zip_code, date_of_birth,
                             account_open_date.
            - "cards": Contains card information.
                Key columns: card_id (PK), customer_id (FK), card_number,
                             card_type, card_provider, issue_date, expiry_date,
                             cvv, credit_limit.
            - "transactions": Contains transaction records.
                Key columns: transaction_id (PK), card_id (FK), transaction_date,
                             transaction_time, amount, merchant_id (FK), location,
                             transaction_type, is_fraud.
            - "merchants": Contains merchant information.
                Key columns: merchant_id (PK), merchant_name, merchant_category,
                             city, state, country.

    Example:
        >>> import datetime
        >>> from featuretools.demo.banking import load_banking_data
        >>> data = load_banking_data(n_customers=10,
        ...                          n_cards_per_customer=1,
        ...                          n_transactions_per_card=5,
        ...                          n_merchants=3,
        ...                          start_date=datetime.date(2024, 1, 1),
        ...                          end_date=datetime.date(2024, 1, 31),
        ...                          fraud_percentage=0.1)
        >>> print(data["customers"].head())
        >>> print(data["cards"].shape)
        >>> print(data["transactions"][["amount", "is_fraud"]].head())
    """
    customers_data = []
    cards_data = []
    transactions_data = []
    merchants_data = []

    # Generate Merchants data
    # Loop to create specified number of unique merchants
    for i in range(n_merchants):
        merchants_data.append({
            "merchant_id": i,
            "merchant_name": fake.company(),
            "merchant_category": fake.bs(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "country": fake.country()
        })
    merchants_df = pd.DataFrame(merchants_data).astype(merchants_schema)

    customer_id_counter = 0
    card_id_counter = 0
    transaction_id_counter = 0

    # Main loop to generate customers and their associated cards and transactions
    for _ in range(n_customers):
        # Generate Customer data
        # Account open date is randomized within the global start and end dates
        account_open_date = fake.date_between_dates(date_start=start_date, date_end=end_date)
        # Date of birth is set to ensure customer is at least 18 years old and not older than 100
        # relative to their account_open_date.
        dob_start_limit = account_open_date - datetime.timedelta(days=100*365)
        dob_end_limit = account_open_date - datetime.timedelta(days=18*365)
        # Ensure dob_start is not before a minimum possible birth year if start_date is very early
        min_birth_year_date = datetime.date(start_date.year - 100, start_date.month, start_date.day) if start_date else None # Avoid issues with very old start_date
        actual_dob_start = max(dob_start_limit, min_birth_year_date) if min_birth_year_date else dob_start_limit

        if actual_dob_start > dob_end_limit : # Fallback if calculated dates are problematic (e.g. very short overall range)
            actual_dob_start = dob_end_limit - datetime.timedelta(days=1*365) # Ensure at least 1 year span for DOB generation

        date_of_birth = fake.date_between_dates(date_start=actual_dob_start, date_end=dob_end_limit)

        customer_info = {
            "customer_id": customer_id_counter,
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.street_address(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "zip_code": fake.zipcode(),
            "date_of_birth": date_of_birth,
            "account_open_date": account_open_date
        }
        customers_data.append(customer_info)

        for _ in range(n_cards_per_customer):
            # Generate Card data for the current customer
            # Card issue date is after account open date and within the global end date
            issue_date = fake.date_between_dates(date_start=account_open_date, date_end=end_date)
            # Card expiry date is set 2-5 years after the issue date
            expiry_date = fake.date_between_dates(date_start=issue_date + datetime.timedelta(days=365*2),
                                                 date_end=issue_date + datetime.timedelta(days=365*5))
            card_type = random.choice(["debit", "credit"])
            card_info = {
                "card_id": card_id_counter,
                "customer_id": customer_id_counter,
                "card_number": fake.credit_card_number(),
                "card_type": card_type,
                "card_provider": fake.credit_card_provider(),
                "issue_date": issue_date,
                "expiry_date": expiry_date,
                "cvv": fake.credit_card_security_code(),
                "credit_limit": round(random.uniform(1000, 20000), 2) if card_type == "credit" else 0.0
            }
            cards_data.append(card_info)

            # Generate Transactions for the current card
            for _ in range(n_transactions_per_card):
                # Determine valid date range for a transaction:
                # Must be after card issue_date and global start_date.
                # Must be before card expiry_date and global end_date.
                effective_transaction_start_date = max(start_date, issue_date)
                effective_transaction_end_date = min(end_date, expiry_date)

                # If the valid date range is impossible (e.g., card expires before it can be used), skip transaction generation.
                if effective_transaction_start_date > effective_transaction_end_date:
                    continue # No valid window for transactions with this card.

                transaction_date = fake.date_between_dates(date_start=effective_transaction_start_date, date_end=effective_transaction_end_date)
                transaction_time = fake.time_object() # Generates a datetime.time object

                # Select a random merchant for the transaction
                # Ensure merchants_data is not empty to avoid error with random.choice
                if not merchants_data: # Should not happen if n_merchants > 0
                    # Handle case with no merchants, perhaps by skipping transaction or logging a warning
                    # For now, we assume n_merchants > 0 as per typical usage.
                    # If it can be 0, this part might need adjustment or transactions shouldn't be generated.
                    pass # Or raise error, or assign a dummy merchant_id like -1
                merchant = random.choice(merchants_data)

                is_fraud = False
                transaction_location = f"{merchant['city']}, {merchant['state']}, {merchant['country']}" # Default location is merchant's location
                amount = round(random.uniform(5, 1000), 2) # Standard transaction amount range

                # Logic to introduce fraudulent transactions
                if random.random() < fraud_percentage:
                    is_fraud = True
                    # Select a type of fraud to simulate
                    fraud_type = random.choice(["large_amount", "unusual_location"])
                    # Future fraud types: "rapid_succession_diff_loc", "new_merchant_category"

                    if fraud_type == "large_amount":
                        # Simulate an unusually large transaction amount
                        amount = round(random.uniform(card_info.get("credit_limit", 1000) * 0.8,
                                                     card_info.get("credit_limit", 1000) * 2 + 5000), 2) # Higher than usual, possibly over limit
                        if amount <= 1000: # Ensure it's actually larger than normal upper bound
                            amount = round(random.uniform(1001, 20000), 2)

                    elif fraud_type == "unusual_location":
                        # Simulate a transaction from a different country than the customer's primary location
                        original_customer_country = "" # Assume we can get this from customer_info if available/relevant
                        # For simplicity, let's use the merchant's country as a base and pick a different one
                        # This simulates the card being used in an unexpected country.
                        # A more sophisticated approach would be to check against customer's country from customer_info.
                        fraud_country = fake.country()
                        while fraud_country == merchant['country']: # Ensure it's a different country
                            fraud_country = fake.country()
                        transaction_location = f"{fake.city()}, {fake.state_abbr()}, {fraud_country}"

                    # Note: More complex fraud patterns like "rapid_succession_diff_loc"
                    # (transactions close in time but far apart geographically) or
                    # "new_merchant_category" (a customer suddenly using a very different merchant type)
                    # would require access to the customer's transaction history or more stateful generation.

                transaction_info = {
                    "transaction_id": transaction_id_counter,
                    "card_id": card_id_counter,
                    "transaction_date": transaction_date,
                    "transaction_time": transaction_time,
                    "amount": amount,
                    "merchant_id": merchant["merchant_id"],
                    "location": f"{merchant['city']}, {merchant['state']}, {merchant['country']}",
                    "transaction_type": random.choice(["purchase", "withdrawal", "transfer"]),
                    "is_fraud": is_fraud
                }
                transactions_data.append(transaction_info)
                transaction_id_counter += 1
            card_id_counter += 1
        customer_id_counter += 1

    customers_df = pd.DataFrame(customers_data).astype(customers_schema)
    cards_df = pd.DataFrame(cards_data).astype(cards_schema)

    if not transactions_data:
        # Create empty DataFrame with correct columns and dtypes if no transactions were generated
        transactions_df = pd.DataFrame(columns=transactions_schema.keys()).astype(transactions_schema)
    else:
        transactions_df = pd.DataFrame(transactions_data).astype(transactions_schema)

    # Convert date/time columns after DataFrame creation for robustness
    customers_df['date_of_birth'] = pd.to_datetime(customers_df['date_of_birth']).dt.date
    customers_df['account_open_date'] = pd.to_datetime(customers_df['account_open_date']).dt.date
    cards_df['issue_date'] = pd.to_datetime(cards_df['issue_date']).dt.date
    cards_df['expiry_date'] = pd.to_datetime(cards_df['expiry_date']).dt.date

    # Only convert transaction dates if DataFrame is not empty
    if not transactions_df.empty:
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date']).dt.date
        # transactions_df['transaction_time'] = pd.to_datetime(transactions_df['transaction_time'], format='%H:%M:%S').dt.time # Ensure this is correct based on fake.time_object() output
    # else:
        # If transactions_df is empty, transaction_time column might not exist yet for conversion
        # or it's already correctly typed as empty 'object' from the schema based empty DF creation.
        # We ensure 'transaction_time' is of object type if it exists and df is empty,
        # which should be the case if initialized from schema.
        # if 'transaction_time' in transactions_df.columns:
        #    transactions_df['transaction_time'] = transactions_df['transaction_time'].astype(object)


    return {
        "customers": customers_df,
        "cards": cards_df,
        "transactions": transactions_df,
        "merchants": merchants_df
    }


if __name__ == '__main__':
    # Constants for the main example execution
    N_CUSTOMERS = 5
    N_CARDS_PER_CUSTOMER = 1
    N_TRANSACTIONS_PER_CARD = 10
    N_MERCHANTS = 3
    START_DATE = datetime.date(2023, 1, 1)
    END_DATE = datetime.date(2023, 3, 31) # Shorter period for readable example
    FRAUD_PERCENTAGE = 0.15 # Slightly higher fraud for visibility in small dataset

    # Detailed example of using load_banking_data and exploring its output.
    # This also serves as a runnable script to check data generation.

    print(f"Generating banking data with: {N_CUSTOMERS} customers, "
          f"{N_CARDS_PER_CUSTOMER} cards/customer, "
          f"{N_TRANSACTIONS_PER_CARD} transactions/card, "
          f"{N_MERCHANTS} merchants.")
    print(f"Date range from {START_DATE} to {END_DATE}.")
    print(f"Target fraud percentage: {FRAUD_PERCENTAGE*100}%.")

    banking_data = load_banking_data(
        n_customers=N_CUSTOMERS,
        n_cards_per_customer=N_CARDS_PER_CUSTOMER,
        n_transactions_per_card=N_TRANSACTIONS_PER_CARD,
        n_merchants=N_MERCHANTS,
        start_date=START_DATE,
        end_date=END_DATE,
        fraud_percentage=FRAUD_PERCENTAGE
    )

    print("\n--- Sample of Generated Data ---")

    for table_name, df in banking_data.items():
        print(f"\n{table_name.capitalize()} DataFrame (Shape: {df.shape}):")
        print(df.head())
        # print(df.info()) # Uncomment for full schema and dtype info

    # Verify actual fraud percentage
    transactions_df = banking_data["transactions"]
    if not transactions_df.empty:
        actual_fraud_percentage = transactions_df["is_fraud"].mean() * 100
        print(f"\nActual fraud percentage in transactions: {actual_fraud_percentage:.2f}%")
    else:
        print("\nNo transactions generated.")

    # --- Optional: Creating a Featuretools EntitySet ---
    # Note: This requires Featuretools to be installed (pip install featuretools)
    try:
        import featuretools as ft
        print("\n--- Creating Featuretools EntitySet ---")

        es = ft.EntitySet(id="banking_demo")

        # Add entities (tables) - ensure correct time_index and index are specified
        es = es.add_dataframe(
            dataframe_name="customers",
            dataframe=banking_data["customers"],
            index="customer_id",
            time_index="account_open_date"
        )

        es = es.add_dataframe(
            dataframe_name="cards",
            dataframe=banking_data["cards"],
            index="card_id",
            time_index="issue_date"
            # logical_types={"card_number": "Categorical"} # Example of specifying logical type
        )

        es = es.add_dataframe(
            dataframe_name="merchants",
            dataframe=banking_data["merchants"],
            index="merchant_id"
            # logical_types={"merchant_category": "Categorical"}
        )

        # Transactions need careful handling of time_index if transaction_time is separate
        # For now, using transaction_date as time_index.
        # If transaction_time is critical, it might need to be combined with transaction_date
        # into a single datetime column before adding to EntitySet for proper time handling.
        transactions_with_datetime = banking_data["transactions"].copy()
        # Combine date and time for a full timestamp - required by Featuretools for time_index
        # This assumes transaction_time is datetime.time, if it's string, parse it first.
        # transactions_with_datetime["transaction_datetime"] = transactions_with_datetime.apply(
        # lambda x: datetime.datetime.combine(x.transaction_date, x.transaction_time), axis=1
        # )

        es = es.add_dataframe(
            dataframe_name="transactions",
            dataframe=transactions_with_datetime, # Use the one with combined datetime if created
            index="transaction_id",
            time_index="transaction_date" # Or "transaction_datetime" if combined
            # logical_types={"transaction_type": "Categorical", "merchant_category_at_transaction": "Categorical"} # Example
        )

        # Define relationships between entities
        es = es.add_relationship("customers", "customer_id", "cards", "customer_id")
        es = es.add_relationship("cards", "card_id", "transactions", "card_id")
        es = es.add_relationship("merchants", "merchant_id", "transactions", "merchant_id")

        print("\nEntitySet created successfully:")
        print(es)

        # Example: Run Deep Feature Synthesis (DFS)
        # print("\n--- Running Deep Feature Synthesis (DFS) example ---")
        # feature_matrix, feature_defs = ft.dfs(
        # entityset=es,
        # target_dataframe_name="customers",
        #     agg_primitives=["mean", "sum", "count", "mode"],
        #     trans_primitives=["month", "weekday", "isin"],
        #     max_depth=2
        # )
        # print("\nFeature Matrix (Customers):")
        # print(feature_matrix.head())

    except ImportError:
        print("\nFeaturetools library not found. Skipping EntitySet creation and DFS example.")
    except Exception as e:
        print(f"\nError creating EntitySet or running DFS: {e}")
        print("Please ensure Featuretools is installed and data is compatible.")

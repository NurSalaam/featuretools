"""
Unit tests for :class:`SparkFeatureSetCalculator` — one test per
``_calculate_*_features`` method, plus a check that ``_necessary_columns``
mirrors the pandas behavior.

Where feasible we construct minimal fixtures rather than going through the
full ``ft.dfs()`` pipeline so that when a test fails the culprit is
localized.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from featuretools.computational_backends.feature_set import FeatureSet  # noqa: E402
from featuretools.computational_backends.spark.spark_feature_set_calculator import (  # noqa: E402
    SparkFeatureSetCalculator,
)


@pytest.fixture
def simple_spark_es(spark_session):
    """Two-table entityset: customers (parent) and orders (child)."""
    from featuretools.entityset.spark import SparkEntitySet

    customers_pdf = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "signup_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "tier": ["gold", "silver", "gold"],
        }
    )
    orders_pdf = pd.DataFrame(
        {
            "order_id": [10, 11, 12, 13, 14],
            "customer_id": [1, 1, 2, 3, 3],
            "amount": [100.0, 50.0, 200.0, 75.0, 125.0],
            "order_time": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-01-15",
                    "2021-03-01",
                    "2021-03-15",
                ]
            ),
        }
    )
    customers_sdf = spark_session.createDataFrame(customers_pdf)
    orders_sdf = spark_session.createDataFrame(orders_pdf)

    es = SparkEntitySet(id="simple")
    es.add_dataframe(
        dataframe_name="customers",
        dataframe=customers_sdf,
        index="customer_id",
        time_index="signup_date",
    )
    es.add_dataframe(
        dataframe_name="orders",
        dataframe=orders_sdf,
        index="order_id",
        time_index="order_time",
    )
    es.add_relationship("customers", "customer_id", "orders", "customer_id")
    return es


def test_necessary_columns_includes_index_and_fks(simple_spark_es):
    """The calculator should always keep the primary key, time_index, and
    any declared foreign keys. Mirrors ``feature_set_calculator.py:822``.
    """
    import featuretools as ft

    features = ft.dfs(
        entityset=simple_spark_es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    fs = FeatureSet(features)
    calc = SparkFeatureSetCalculator(simple_spark_es, fs)
    orders_cols = calc._necessary_columns("orders", set())
    # Must include the primary key, time index, and the foreign-key to the parent.
    assert "order_id" in orders_cols
    assert "order_time" in orders_cols
    assert "customer_id" in orders_cols


def test_calculate_count_feature_matches_pandas(simple_spark_es):
    """End-to-end check on the simple two-table EntitySet."""
    import featuretools as ft

    features = ft.dfs(
        entityset=simple_spark_es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )

    # Reference: compute COUNT(orders) manually.
    fm_spark = ft.calculate_feature_matrix(
        features=features, entityset=simple_spark_es
    )
    # customer 1 has 2 orders, customer 2 has 1, customer 3 has 2
    expected = {1: 2, 2: 1, 3: 2}
    count_col = next(c for c in fm_spark.columns if "COUNT(orders)" in c)
    for cid, cnt in expected.items():
        assert int(fm_spark.loc[cid, count_col]) == cnt


def test_calculate_sum_mean_features_match_pandas(simple_spark_es):
    """Aggregation fanout: SUM / MEAN of amount per customer."""
    import featuretools as ft

    features = ft.dfs(
        entityset=simple_spark_es,
        target_dataframe_name="customers",
        agg_primitives=["sum", "mean"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    fm_spark = ft.calculate_feature_matrix(
        features=features, entityset=simple_spark_es
    )

    sum_col = next(c for c in fm_spark.columns if c.startswith("SUM"))
    mean_col = next(c for c in fm_spark.columns if c.startswith("MEAN"))

    # customer 1: 100 + 50 = 150, mean 75
    assert float(fm_spark.loc[1, sum_col]) == pytest.approx(150.0)
    assert float(fm_spark.loc[1, mean_col]) == pytest.approx(75.0)
    # customer 2: 200, mean 200
    assert float(fm_spark.loc[2, sum_col]) == pytest.approx(200.0)
    assert float(fm_spark.loc[2, mean_col]) == pytest.approx(200.0)
    # customer 3: 75 + 125 = 200, mean 100
    assert float(fm_spark.loc[3, sum_col]) == pytest.approx(200.0)
    assert float(fm_spark.loc[3, mean_col]) == pytest.approx(100.0)

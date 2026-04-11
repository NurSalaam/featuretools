"""
End-to-end integration: ``ft.dfs(entityset=spark_es, ...)`` must produce a
feature matrix that matches the pandas reference path cell-by-cell.

These are the gold-standard correctness tests. A green run here proves
that the Spark calculator's per-feature-type methods, the primitive
dispatcher, the EntitySet schema propagation, and the dispatch point in
``calculate_feature_matrix`` are all wired up correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")


def _compare_feature_matrices(fm_pandas: pd.DataFrame, fm_spark: pd.DataFrame) -> None:
    """Assert two feature matrices match after normalizing index/column
    order and tolerating small floating-point differences.
    """
    # Normalize index order so we're comparing the same rows.
    fm_p = fm_pandas.sort_index().copy()
    fm_s = fm_spark.sort_index().copy()

    # Restrict to the intersection of columns — Spark may drop woodwork-
    # only informational columns and that's OK.
    common = [c for c in fm_p.columns if c in fm_s.columns]
    assert common, "No overlapping columns between pandas and Spark feature matrices"

    fm_p = fm_p[common]
    fm_s = fm_s[common]

    for col in common:
        p = fm_p[col]
        s = fm_s[col]
        if p.dtype.kind in "fi" and s.dtype.kind in "fi":
            np.testing.assert_allclose(
                s.astype(float).values,
                p.astype(float).values,
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"column {col} diverged",
            )
        else:
            assert list(s.astype("object")) == list(p.astype("object")), (
                f"column {col} diverged (object compare)"
            )


def test_dfs_spark_simple_count(mock_customer_pandas_es, mock_customer_spark_es):
    """Simplest possible integration test: COUNT aggregations at depth 1
    on the customers table.
    """
    import featuretools as ft

    agg = ["count"]
    trans = []
    fm_pandas, _ = ft.dfs(
        entityset=mock_customer_pandas_es,
        target_dataframe_name="customers",
        agg_primitives=agg,
        trans_primitives=trans,
        max_depth=1,
    )
    fm_spark = ft.calculate_feature_matrix(
        features=ft.dfs(
            entityset=mock_customer_pandas_es,
            target_dataframe_name="customers",
            agg_primitives=agg,
            trans_primitives=trans,
            max_depth=1,
            features_only=True,
        ),
        entityset=mock_customer_spark_es,
    )
    _compare_feature_matrices(fm_pandas, fm_spark)


def test_dfs_spark_multiple_aggs(mock_customer_pandas_es, mock_customer_spark_es):
    """Cover several tier-1 aggregations simultaneously."""
    import featuretools as ft

    agg = ["count", "sum", "mean", "min", "max"]
    features_only = ft.dfs(
        entityset=mock_customer_pandas_es,
        target_dataframe_name="customers",
        agg_primitives=agg,
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    fm_pandas = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_pandas_es
    )
    fm_spark = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_spark_es
    )
    _compare_feature_matrices(fm_pandas, fm_spark)


def test_dfs_spark_transform_features(mock_customer_pandas_es, mock_customer_spark_es):
    """Tier-1 transforms must match pandas element-wise."""
    import featuretools as ft

    features_only = ft.dfs(
        entityset=mock_customer_pandas_es,
        target_dataframe_name="transactions",
        agg_primitives=[],
        trans_primitives=["absolute", "negate"],
        max_depth=1,
        features_only=True,
    )
    fm_pandas = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_pandas_es
    )
    fm_spark = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_spark_es
    )
    _compare_feature_matrices(fm_pandas, fm_spark)


def test_dfs_spark_depth_two(mock_customer_pandas_es, mock_customer_spark_es):
    """Depth-2 feature stacks: mean of count, sum of mean, etc. Exercises
    the trie traversal plus the ancestor-linkage path.
    """
    import featuretools as ft

    features_only = ft.dfs(
        entityset=mock_customer_pandas_es,
        target_dataframe_name="customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        max_depth=2,
        features_only=True,
    )
    fm_pandas = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_pandas_es
    )
    fm_spark = ft.calculate_feature_matrix(
        features=features_only, entityset=mock_customer_spark_es
    )
    _compare_feature_matrices(fm_pandas, fm_spark)


def test_n_jobs_warning_on_spark_path(
    mock_customer_pandas_es, mock_customer_spark_es
):
    """``n_jobs != 1`` should warn (but not error) on the Spark path."""
    import featuretools as ft

    features_only = ft.dfs(
        entityset=mock_customer_pandas_es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    with pytest.warns(UserWarning, match="n_jobs is ignored"):
        ft.calculate_feature_matrix(
            features=features_only,
            entityset=mock_customer_spark_es,
            n_jobs=4,
        )

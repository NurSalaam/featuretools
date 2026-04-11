"""
Mapping-table round-trip tests.

For each primitive registered in ``SPARK_AGG_MAP`` / ``SPARK_TRANSFORM_MAP``
we:
  1. Generate a small synthetic DataFrame.
  2. Compute the reference value using the primitive's pandas function.
  3. Compute the Spark value using the mapping-table expression.
  4. Assert they match within floating-point tolerance.

These tests are the backstop for the central registration pattern — if
someone changes the Spark mapping or a primitive adds an incompatible
signature, a test here should fail loudly.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from featuretools.computational_backends.spark.spark_primitive_dispatch import (  # noqa: E402
    SPARK_AGG_MAP,
    SPARK_TRANSFORM_MAP,
    register_builtin_mappings,
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_registered():
    register_builtin_mappings()


@pytest.fixture
def numeric_df(spark_session):
    pdf = pd.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "grp": ["a"] * 5 + ["b"] * 5,
            "flag": [True, False, True, True, False, True, True, False, True, False],
        }
    )
    return spark_session.createDataFrame(pdf), pdf


# ----------------------------------------------------------------------
# Aggregation primitives — parametrized round-trip.
# ----------------------------------------------------------------------
AGG_TEST_CASES = [
    ("Count", lambda s: s.count()),
    ("Sum", lambda s: s.sum()),
    ("Mean", lambda s: s.mean()),
    ("Min", lambda s: s.min()),
    ("Max", lambda s: s.max()),
    ("Std", lambda s: s.std()),
    ("NumUnique", lambda s: s.nunique()),
]


@pytest.mark.parametrize("name,reference", AGG_TEST_CASES)
def test_spark_agg_matches_pandas(numeric_df, name, reference):
    sdf, pdf = numeric_df
    # Find the registered primitive class by name.
    cls = next(
        (c for c in SPARK_AGG_MAP if c.__name__ == name or c.__name__.replace("Primitive", "") == name),
        None,
    )
    if cls is None:
        pytest.skip(f"{name} not registered in SPARK_AGG_MAP")

    spark_fn = SPARK_AGG_MAP[cls]
    spark_val = sdf.agg(spark_fn("val").alias("out")).collect()[0]["out"]
    pandas_val = reference(pdf["val"])

    if spark_val is None or pandas_val is None:
        assert spark_val is None and pandas_val is None
        return
    assert math.isclose(float(spark_val), float(pandas_val), rel_tol=1e-9, abs_tol=1e-9)


def test_spark_percent_true(numeric_df):
    from featuretools.primitives.standard.aggregation.percent_true import PercentTrue

    sdf, pdf = numeric_df
    if PercentTrue not in SPARK_AGG_MAP:
        pytest.skip("PercentTrue not registered")
    spark_fn = SPARK_AGG_MAP[PercentTrue]
    spark_val = sdf.agg(spark_fn("flag").alias("out")).collect()[0]["out"]
    pandas_val = pdf["flag"].mean()
    assert math.isclose(float(spark_val), float(pandas_val), rel_tol=1e-9)


# ----------------------------------------------------------------------
# Transform primitives — parametrized element-wise round-trip.
# ----------------------------------------------------------------------
TRANS_TEST_CASES = [
    ("Absolute", lambda s: s.abs()),
    ("NaturalLogarithm", lambda s: s.apply(lambda v: math.log(v) if v > 0 else None)),
    ("Negate", lambda s: -s),
    ("SquareRoot", lambda s: s.apply(lambda v: math.sqrt(v) if v >= 0 else None)),
]


@pytest.mark.parametrize("name,reference", TRANS_TEST_CASES)
def test_spark_transform_matches_pandas(numeric_df, name, reference):
    from pyspark.sql import functions as F

    sdf, pdf = numeric_df
    cls = next((c for c in SPARK_TRANSFORM_MAP if c.__name__ == name), None)
    if cls is None:
        pytest.skip(f"{name} not registered in SPARK_TRANSFORM_MAP")

    spark_fn = SPARK_TRANSFORM_MAP[cls]
    result_col = spark_fn([F.col("val")])
    spark_result = sdf.select(result_col.alias("out")).toPandas()["out"]
    pandas_result = reference(pdf["val"])

    # Convert to numeric and compare element-wise.
    for s, p in zip(spark_result, pandas_result):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            assert s is None or (isinstance(s, float) and math.isnan(s))
            continue
        assert math.isclose(float(s), float(p), rel_tol=1e-9, abs_tol=1e-9)


def test_weekday_matches_pandas_convention(spark_session):
    """Pandas uses 0=Monday..6=Sunday. We wrap Spark's 1=Sunday..7=Saturday
    convention to match. Regression test for the conversion.
    """
    from featuretools.primitives.standard.transform.datetime.weekday import Weekday
    from pyspark.sql import functions as F

    if Weekday not in SPARK_TRANSFORM_MAP:
        pytest.skip("Weekday not registered")

    # 2024-01-01 is a Monday → pandas weekday == 0
    pdf = pd.DataFrame(
        {
            "d": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                ]
            )
        }
    )
    sdf = spark_session.createDataFrame(pdf)
    expr = SPARK_TRANSFORM_MAP[Weekday]([F.col("d")])
    result = sdf.select(expr.alias("wd")).toPandas()["wd"].tolist()
    expected = pdf["d"].dt.weekday.tolist()
    assert result == expected

"""
Low-level helpers for the Spark DFS backend.

Handles:
- Logical-type → Spark ``DataType`` conversion.
- Python default value → Spark literal conversion.
- Cheap-ish broadcast sizing heuristic (so we don't hint a broadcast for a
  dimension table that's actually large).

Kept deliberately small: anything that requires feature-level reasoning
belongs in ``spark_primitive_dispatch`` or ``spark_feature_set_calculator``.
"""
from __future__ import annotations

from typing import Any

# Conservative threshold for F.broadcast hints — matches the default of
# ``spark.sql.autoBroadcastJoinThreshold`` (10 MB) assuming ~100 bytes/row.
# Above this row count we let Spark's own optimizer decide.
DEFAULT_BROADCAST_ROW_THRESHOLD = 100_000


def logical_type_to_spark_type(logical_type: str):
    """Map a ``SparkLogicalSchema`` logical-type tag to a Spark ``DataType``.

    Returns ``None`` if the tag is unrecognized, so callers can fall back to
    letting Spark infer the type at DataFrame construction time.
    """
    from pyspark.sql import types as T

    table = {
        "Integer": T.LongType(),
        "IntegerNullable": T.LongType(),
        "Double": T.DoubleType(),
        "Boolean": T.BooleanType(),
        "BooleanNullable": T.BooleanType(),
        "Datetime": T.TimestampType(),
        "Timedelta": T.DayTimeIntervalType(),
        "Categorical": T.StringType(),
        "Ordinal": T.StringType(),
        "NaturalLanguage": T.StringType(),
        "EmailAddress": T.StringType(),
        "URL": T.StringType(),
        "PhoneNumber": T.StringType(),
        "PostalCode": T.StringType(),
        "CountryCode": T.StringType(),
        "IPAddress": T.StringType(),
    }
    return table.get(logical_type)


def return_type_to_spark_type(column_schema) -> "pyspark.sql.types.DataType":  # noqa: F821
    """Convert a woodwork ``ColumnSchema`` return_type (as carried on
    stdlib primitives) into a Spark ``DataType``.

    We only need a rough mapping for primitive outputs — the actual value
    will be cast by Spark at materialization time.
    """
    from pyspark.sql import types as T

    if column_schema is None:
        return T.DoubleType()

    # woodwork ColumnSchema exposes a .logical_type class; we dispatch on its
    # name to avoid importing every woodwork logical type.
    lt = getattr(column_schema, "logical_type", None)
    name = type(lt).__name__ if lt is not None else None
    if name is None:
        # Fall back on semantic tags.
        tags = getattr(column_schema, "semantic_tags", set()) or set()
        if "numeric" in tags:
            return T.DoubleType()
        if "category" in tags:
            return T.StringType()
        return T.StringType()

    table = {
        "Integer": T.LongType(),
        "IntegerNullable": T.LongType(),
        "Age": T.LongType(),
        "AgeNullable": T.LongType(),
        "Double": T.DoubleType(),
        "AgeFractional": T.DoubleType(),
        "Boolean": T.BooleanType(),
        "BooleanNullable": T.BooleanType(),
        "Datetime": T.TimestampType(),
        "Timedelta": T.DayTimeIntervalType(),
        "Categorical": T.StringType(),
        "Ordinal": T.StringType(),
        "NaturalLanguage": T.StringType(),
        "EmailAddress": T.StringType(),
        "URL": T.StringType(),
        "PhoneNumber": T.StringType(),
        "PostalCode": T.StringType(),
        "CountryCode": T.StringType(),
        "IPAddress": T.StringType(),
    }
    return table.get(name, T.DoubleType())


def default_to_spark_literal(value: Any):
    """Convert a Python default value to a Spark ``F.lit()``, handling NaN,
    None, and numpy scalars."""
    import numpy as np
    from pyspark.sql import functions as F

    if value is None:
        return F.lit(None)
    if isinstance(value, float) and np.isnan(value):
        return F.lit(None)
    if isinstance(value, np.generic):
        return F.lit(value.item())
    return F.lit(value)


def should_broadcast(spark_df, row_threshold: int = DEFAULT_BROADCAST_ROW_THRESHOLD):
    """Return True if we should hint Spark to broadcast ``spark_df`` in a
    join. Uses a cheap approximate count. For very large tables, ``.count()``
    would be expensive — prefer metadata if available.

    First-cut heuristic: if the DataFrame has a cached ``numRows`` in its
    execution stats, use it; otherwise default to True for tables that
    *look* small by schema heuristics and rely on Spark to rewrite the plan
    if the broadcast threshold is exceeded.
    """
    try:
        # Spark 3.x exposes stats via the logical plan.
        stats = spark_df._jdf.queryExecution().optimizedPlan().stats()
        size_bytes = stats.sizeInBytes()
        # 10 MB default threshold
        return size_bytes < 10 * 1024 * 1024
    except Exception:
        # If we can't figure out a size, default to broadcasting small
        # dimension tables. Spark will ignore the hint if the actual size
        # exceeds autoBroadcastJoinThreshold.
        return True

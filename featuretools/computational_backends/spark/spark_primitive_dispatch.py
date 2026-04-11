"""
Primitive dispatcher for the native Spark DFS backend.

This module decides, for each Featuretools primitive, how to compute it on
Spark:

- **Tier 1 — Spark-SQL expression.** The primitive class is registered in
  one of the ``SPARK_*_MAP`` tables below and we call the mapped
  ``pyspark.sql.functions`` callable directly. Zero Python round-trips.
  Fastest.
- **Tier 2 — ``pandas_udf``.** Default fallback for any primitive whose
  ``get_function()`` returns a callable that accepts a ``pd.Series`` and
  returns a ``pd.Series`` (or a scalar, for aggregation UDFs). Arrow-based,
  vectorized. A one-time warning is emitted per primitive class per
  session so users know they're not on the fast path.
- **Tier 3 — Python UDF.** Only used when the primitive is explicitly
  flagged ``spark_rowwise_only = True``. Emits a one-time warning.

Registration of stdlib primitives happens centrally in
:func:`register_builtin_mappings`, which is called automatically when this
module is imported. We deliberately avoid per-primitive edits so the
reviewable surface stays small.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger("featuretools.computational_backends.spark.dispatch")

# Module-level state: set of primitive classes we've already warned about
# so users see each fallback at most once per session.
_WARNED_PRIMITIVES: set = set()


# ----------------------------------------------------------------------
# Mapping tables — populated by register_builtin_mappings()
# ----------------------------------------------------------------------
# Each value is a callable that receives `list[Column]` (transform) or
# `col_name: str` (aggregation/groupby) and returns a Spark Column.
SPARK_AGG_MAP: Dict[Type, Callable] = {}
SPARK_TRANSFORM_MAP: Dict[Type, Callable] = {}
SPARK_GROUPBY_TRANS_MAP: Dict[Type, Callable] = {}


_BUILTINS_REGISTERED = False


def register_builtin_mappings() -> None:
    """Populate SPARK_AGG_MAP / SPARK_TRANSFORM_MAP / SPARK_GROUPBY_TRANS_MAP
    with the first-cut set of stdlib primitives.

    We import each primitive class lazily and bind it to a
    ``pyspark.sql.functions`` callable. Only the ``pyspark`` imports are
    lazy — the featuretools imports are top-level so registration fails
    loud if a primitive has moved.

    Idempotent: safe to call multiple times.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    try:
        from pyspark.sql import Window, functions as F
    except ImportError:
        # No pyspark installed — leave the maps empty. The SparkEntitySet
        # import guard in featuretools/entityset/__init__.py means we
        # shouldn't actually be running the Spark path without pyspark, but
        # we don't want merely importing the module to blow up.
        _BUILTINS_REGISTERED = True
        return

    # ------------------------------------------------------------------
    # Aggregation primitives
    # ------------------------------------------------------------------
    from featuretools.primitives.standard.aggregation.count import Count
    from featuretools.primitives.standard.aggregation.sum_primitive import Sum
    from featuretools.primitives.standard.aggregation.mean import Mean
    from featuretools.primitives.standard.aggregation.min_primitive import Min
    from featuretools.primitives.standard.aggregation.max_primitive import Max
    from featuretools.primitives.standard.aggregation.std import Std
    from featuretools.primitives.standard.aggregation.skew import Skew
    from featuretools.primitives.standard.aggregation.median import Median
    from featuretools.primitives.standard.aggregation.num_unique import NumUnique
    from featuretools.primitives.standard.aggregation.percent_true import PercentTrue
    from featuretools.primitives.standard.aggregation.first import First
    from featuretools.primitives.standard.aggregation.last import Last
    from featuretools.primitives.standard.aggregation.any_primitive import Any as AnyAgg
    from featuretools.primitives.standard.aggregation.all_primitive import All as AllAgg

    SPARK_AGG_MAP[Count] = lambda col: F.count(F.col(col))
    SPARK_AGG_MAP[Sum] = lambda col: F.sum(F.col(col))
    SPARK_AGG_MAP[Mean] = lambda col: F.avg(F.col(col))
    SPARK_AGG_MAP[Min] = lambda col: F.min(F.col(col))
    SPARK_AGG_MAP[Max] = lambda col: F.max(F.col(col))
    SPARK_AGG_MAP[Std] = lambda col: F.stddev_samp(F.col(col))
    SPARK_AGG_MAP[Skew] = lambda col: F.skewness(F.col(col))
    SPARK_AGG_MAP[Median] = lambda col: F.percentile_approx(F.col(col), 0.5)
    SPARK_AGG_MAP[NumUnique] = lambda col: F.countDistinct(F.col(col))
    SPARK_AGG_MAP[PercentTrue] = lambda col: F.avg(F.col(col).cast("double"))
    SPARK_AGG_MAP[First] = lambda col: F.first(F.col(col), ignorenulls=True)
    SPARK_AGG_MAP[Last] = lambda col: F.last(F.col(col), ignorenulls=True)
    # bool_or / bool_and were added in Spark 3.5. If the runtime is older we
    # fall back to casting + max/min on a 0/1 column.
    if hasattr(F, "bool_or"):
        SPARK_AGG_MAP[AnyAgg] = lambda col: F.bool_or(F.col(col))
        SPARK_AGG_MAP[AllAgg] = lambda col: F.bool_and(F.col(col))
    else:
        SPARK_AGG_MAP[AnyAgg] = lambda col: (
            F.max(F.col(col).cast("int")).cast("boolean")
        )
        SPARK_AGG_MAP[AllAgg] = lambda col: (
            F.min(F.col(col).cast("int")).cast("boolean")
        )

    # ------------------------------------------------------------------
    # Transform primitives
    # ------------------------------------------------------------------
    from featuretools.primitives.standard.transform.numeric.absolute import Absolute
    from featuretools.primitives.standard.transform.numeric.natural_logarithm import (
        NaturalLogarithm,
    )
    from featuretools.primitives.standard.transform.numeric.negate import Negate
    from featuretools.primitives.standard.transform.numeric.sine import Sine
    from featuretools.primitives.standard.transform.numeric.cosine import Cosine
    from featuretools.primitives.standard.transform.numeric.tangent import Tangent
    from featuretools.primitives.standard.transform.numeric.square_root import (
        SquareRoot,
    )
    from featuretools.primitives.standard.transform.is_null import IsNull
    from featuretools.primitives.standard.transform.datetime.year import Year
    from featuretools.primitives.standard.transform.datetime.month import Month
    from featuretools.primitives.standard.transform.datetime.day import Day
    from featuretools.primitives.standard.transform.datetime.hour import Hour
    from featuretools.primitives.standard.transform.datetime.minute import Minute
    from featuretools.primitives.standard.transform.datetime.second import Second
    from featuretools.primitives.standard.transform.datetime.weekday import Weekday

    SPARK_TRANSFORM_MAP[Absolute] = lambda cols: F.abs(cols[0])
    SPARK_TRANSFORM_MAP[NaturalLogarithm] = lambda cols: F.ln(cols[0])
    SPARK_TRANSFORM_MAP[Negate] = lambda cols: -cols[0]
    SPARK_TRANSFORM_MAP[Sine] = lambda cols: F.sin(cols[0])
    SPARK_TRANSFORM_MAP[Cosine] = lambda cols: F.cos(cols[0])
    SPARK_TRANSFORM_MAP[Tangent] = lambda cols: F.tan(cols[0])
    SPARK_TRANSFORM_MAP[SquareRoot] = lambda cols: F.sqrt(cols[0])
    SPARK_TRANSFORM_MAP[IsNull] = lambda cols: F.isnull(cols[0])
    SPARK_TRANSFORM_MAP[Year] = lambda cols: F.year(cols[0])
    SPARK_TRANSFORM_MAP[Month] = lambda cols: F.month(cols[0])
    SPARK_TRANSFORM_MAP[Day] = lambda cols: F.dayofmonth(cols[0])
    SPARK_TRANSFORM_MAP[Hour] = lambda cols: F.hour(cols[0])
    SPARK_TRANSFORM_MAP[Minute] = lambda cols: F.minute(cols[0])
    SPARK_TRANSFORM_MAP[Second] = lambda cols: F.second(cols[0])
    # pandas uses 0=Monday..6=Sunday; Spark's dayofweek returns 1=Sunday..7=Saturday.
    # Convert: (dayofweek - 2) mod 7 → pandas weekday.
    SPARK_TRANSFORM_MAP[Weekday] = lambda cols: (
        (F.dayofweek(cols[0]) - F.lit(2)) % F.lit(7)
    )

    # ------------------------------------------------------------------
    # GroupByTransform primitives (window-function-based)
    # ------------------------------------------------------------------
    try:
        from featuretools.primitives.standard.transform.cumulative.cum_sum import CumSum
        from featuretools.primitives.standard.transform.cumulative.cum_count import (
            CumCount,
        )
        from featuretools.primitives.standard.transform.cumulative.cum_max import CumMax
        from featuretools.primitives.standard.transform.cumulative.cum_min import CumMin
        from featuretools.primitives.standard.transform.cumulative.cum_mean import (
            CumMean,
        )

        def _cum_window_builder(spark_fn):
            def _build(col_name: str, partition_col: str, order_col: Optional[str]):
                w = Window.partitionBy(partition_col)
                if order_col is not None:
                    w = w.orderBy(order_col)
                else:
                    w = w.orderBy(F.monotonically_increasing_id())
                w = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
                return spark_fn(F.col(col_name)).over(w)

            return _build

        SPARK_GROUPBY_TRANS_MAP[CumSum] = _cum_window_builder(F.sum)
        SPARK_GROUPBY_TRANS_MAP[CumCount] = _cum_window_builder(F.count)
        SPARK_GROUPBY_TRANS_MAP[CumMax] = _cum_window_builder(F.max)
        SPARK_GROUPBY_TRANS_MAP[CumMin] = _cum_window_builder(F.min)
        SPARK_GROUPBY_TRANS_MAP[CumMean] = _cum_window_builder(F.avg)
    except ImportError:
        # Older featuretools layout — no-op, CumSum etc. will fall through
        # to pandas_udf grouped-map.
        pass

    _BUILTINS_REGISTERED = True


# ----------------------------------------------------------------------
# Tier 1 detection
# ----------------------------------------------------------------------
def _is_tier1_transform(primitive) -> bool:
    return type(primitive) in SPARK_TRANSFORM_MAP or getattr(
        primitive, "spark_compatible", False
    )


def _is_tier1_agg(primitive) -> bool:
    return type(primitive) in SPARK_AGG_MAP or getattr(
        primitive, "spark_compatible", False
    )


def _is_tier1_groupby(primitive) -> bool:
    return type(primitive) in SPARK_GROUPBY_TRANS_MAP or getattr(
        primitive, "spark_compatible", False
    )


def _warn_fallback(primitive, tier: str) -> None:
    key = (type(primitive), tier)
    if key in _WARNED_PRIMITIVES:
        return
    _WARNED_PRIMITIVES.add(key)
    warnings.warn(
        f"Primitive {type(primitive).__name__}: no Spark-native {tier} "
        f"implementation registered — falling back to pandas_udf. Register "
        f"it in spark_primitive_dispatch.py or subclass SparkPrimitiveMixin "
        f"for better performance.",
        stacklevel=3,
    )


# ----------------------------------------------------------------------
# Transform dispatch
# ----------------------------------------------------------------------
def transform_feature_expr(feature, input_cols):
    """Return a Spark ``Column`` (or list of Columns for multi-output
    primitives) that computes ``feature`` given pre-fetched input columns.

    Parameters
    ----------
    feature : TransformFeature
        The feature to compute.
    input_cols : list[pyspark.sql.Column]
        One Column per base feature, in order.
    """
    from pyspark.sql import functions as F

    primitive = feature.primitive

    # Tier 1: central mapping table
    if type(primitive) in SPARK_TRANSFORM_MAP:
        fn = SPARK_TRANSFORM_MAP[type(primitive)]
        return fn(input_cols)

    # Tier 1: third-party primitive using SparkPrimitiveMixin
    if getattr(primitive, "spark_compatible", False) and hasattr(
        primitive, "get_spark_function"
    ):
        fn = primitive.get_spark_function()
        return fn(input_cols)

    # Tier 3: rowwise-only Python UDF
    if getattr(primitive, "spark_rowwise_only", False):
        _warn_fallback(primitive, "transform (rowwise)")
        return _wrap_python_udf(primitive, input_cols)

    # Tier 2: vectorized pandas_udf (default fallback)
    _warn_fallback(primitive, "transform")
    return _wrap_pandas_udf_transform(primitive, input_cols)


def aggregation_feature_expr(feature, col_name: Optional[str]):
    """Return a Spark aggregation ``Column`` (for use inside
    ``groupBy().agg(...)``).

    ``col_name`` is the name of the single input column. Multi-column
    aggregation primitives go through the ``pandas_udf`` fallback path.
    """
    from pyspark.sql import functions as F

    primitive = feature.primitive

    # Tier 1: central mapping table
    if type(primitive) in SPARK_AGG_MAP:
        fn = SPARK_AGG_MAP[type(primitive)]
        return fn(col_name)

    # Tier 1: third-party primitive using SparkPrimitiveMixin
    if getattr(primitive, "spark_compatible", False) and hasattr(
        primitive, "get_spark_agg_function"
    ):
        fn = primitive.get_spark_agg_function()
        return fn(col_name)

    # Tier 2: vectorized pandas_udf GROUPED_AGG fallback
    _warn_fallback(primitive, "aggregation")
    return _wrap_pandas_udf_agg(primitive, col_name)


def groupby_transform_feature_expr(
    feature, col_name: str, partition_col: str, order_col: Optional[str]
):
    """Return a window-function ``Column`` for a
    :class:`GroupByTransformFeature`. Only single-input primitives are
    supported via the central map; others fall through to a grouped-map
    ``pandas_udf`` and must be handled upstream (we raise NotImplementedError
    here and let the calculator surface a clearer error).
    """
    primitive = feature.primitive
    if type(primitive) in SPARK_GROUPBY_TRANS_MAP:
        return SPARK_GROUPBY_TRANS_MAP[type(primitive)](
            col_name, partition_col, order_col
        )
    if getattr(primitive, "spark_compatible", False) and hasattr(
        primitive, "get_spark_function"
    ):
        # Custom third-party window primitive: expect a builder callable.
        return primitive.get_spark_function()(col_name, partition_col, order_col)

    raise NotImplementedError(
        f"GroupByTransform primitive {type(primitive).__name__} has no "
        f"Spark implementation. Register it in SPARK_GROUPBY_TRANS_MAP or "
        f"compute this feature via the pandas backend."
    )


# ----------------------------------------------------------------------
# pandas_udf fallback builders
# ----------------------------------------------------------------------
def _wrap_pandas_udf_transform(primitive, input_cols):
    """Wrap a transform primitive's ``get_function()`` in a scalar
    ``pandas_udf`` and invoke it.

    The primitive's ``get_function()`` already expects ``pd.Series`` inputs
    (see :class:`PrimitiveBase.__call__`), so we can pass Arrow-converted
    pandas Series straight through.
    """
    from pyspark.sql import functions as F

    from featuretools.computational_backends.spark.spark_utils import (
        return_type_to_spark_type,
    )

    return_type = return_type_to_spark_type(getattr(primitive, "return_type", None))
    func = primitive.get_function()

    @F.pandas_udf(return_type)  # type: ignore[misc]
    def _udf(*series):
        import pandas as pd  # noqa: F401

        result = func(*series)
        return result

    return _udf(*input_cols)


def _wrap_pandas_udf_agg(primitive, col_name: str):
    """Wrap an aggregation primitive's ``get_function()`` in a
    ``GROUPED_AGG`` ``pandas_udf``.

    The primitive is expected to return a scalar from a ``pd.Series``
    input (matching the contract of AggregationPrimitive). For primitives
    that need the full group's metadata (e.g. ``uses_calc_time``), the
    caller must set ``time_last`` via closure before dispatching.
    """
    from pyspark.sql import functions as F

    from featuretools.computational_backends.spark.spark_utils import (
        return_type_to_spark_type,
    )

    return_type = return_type_to_spark_type(getattr(primitive, "return_type", None))
    func = primitive.get_function()

    @F.pandas_udf(return_type, functionType=F.PandasUDFType.GROUPED_AGG)  # type: ignore[arg-type]
    def _udf(series):
        return func(series)

    return _udf(F.col(col_name))


def _wrap_python_udf(primitive, input_cols):
    """Tier-3 fallback: a per-row Python UDF. Avoid unless the primitive
    explicitly opts in via ``spark_rowwise_only = True``."""
    from pyspark.sql import functions as F

    from featuretools.computational_backends.spark.spark_utils import (
        return_type_to_spark_type,
    )

    return_type = return_type_to_spark_type(getattr(primitive, "return_type", None))
    func = primitive.get_function()

    def _apply(*args):
        return func(*args)

    return F.udf(_apply, return_type)(*input_cols)


# Register on import. This mutates module globals but is idempotent.
register_builtin_mappings()

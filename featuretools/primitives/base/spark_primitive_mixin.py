"""
Public extension point for Spark-native primitive implementations.

Third-party primitive packages can subclass :class:`SparkPrimitiveMixin`
alongside their existing ``TransformPrimitive`` / ``AggregationPrimitive``
subclass to opt into the native Spark execution path. Built-in stdlib
primitives are registered centrally in
:mod:`featuretools.computational_backends.spark.spark_primitive_dispatch`
(no per-file edits), so subclassing is only needed for new/external
primitives.

Example
-------

.. code-block:: python

    from featuretools.primitives import AggregationPrimitive
    from featuretools.primitives.base.spark_primitive_mixin import (
        SparkPrimitiveMixin,
    )
    from pyspark.sql import functions as F

    class MyMean(AggregationPrimitive, SparkPrimitiveMixin):
        name = "my_mean"
        spark_compatible = True

        def get_function(self):
            import pandas as pd
            return pd.Series.mean

        def get_spark_agg_function(self):
            return F.avg

The dispatcher checks for ``spark_compatible`` and the presence of
``get_spark_function`` / ``get_spark_agg_function`` via duck typing, so
strictly subclassing the mixin is optional — it is provided for clarity
and to document the contract.
"""
from __future__ import annotations

from typing import Callable, List


class SparkPrimitiveMixin:
    """Optional mixin that marks a primitive as Spark-native.

    Attributes
    ----------
    spark_compatible : bool
        If True, the Spark dispatcher will call ``get_spark_function()`` /
        ``get_spark_agg_function()`` instead of wrapping the primitive in a
        ``pandas_udf``. Default: False.
    spark_rowwise_only : bool
        If True, the Spark dispatcher will use a Python UDF instead of a
        vectorized ``pandas_udf``. Only set this if your primitive cannot
        accept a ``pd.Series`` (or the ``pandas_udf`` Arrow conversion is
        otherwise unsafe). Default: False.
    """

    spark_compatible: bool = False
    spark_rowwise_only: bool = False

    def get_spark_function(self) -> Callable[[List], "pyspark.sql.Column"]:  # noqa: F821
        """Return a callable ``(list[Column]) -> Column`` for transform
        primitives. Only called when ``spark_compatible`` is True. For
        multi-output primitives, return a ``list[Column]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_spark_function is not implemented. "
            "Either set spark_compatible=False to use the pandas_udf "
            "fallback, or implement this method."
        )

    def get_spark_agg_function(self) -> Callable[[str], "pyspark.sql.Column"]:  # noqa: F821
        """Return a callable ``(col_name: str) -> Column`` for aggregation
        primitives. Only called when ``spark_compatible`` is True.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_spark_agg_function is not "
            "implemented. Either set spark_compatible=False to use the "
            "pandas_udf GROUPED_AGG fallback, or implement this method."
        )

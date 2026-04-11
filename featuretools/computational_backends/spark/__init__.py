"""
Native PySpark execution engine for Featuretools DFS.

This subpackage mirrors the shape of
:mod:`featuretools.computational_backends` but every operation is built on
Spark DataFrame APIs (``groupBy``, ``agg``, ``join``, window functions) with
``pandas_udf`` fallbacks for primitives that lack a Spark-SQL equivalent.

The dispatcher at the top of
:func:`featuretools.computational_backends.calculate_feature_matrix` routes
Spark EntitySets here. Pandas EntitySets are unaffected.
"""
from featuretools.computational_backends.spark.spark_calculate_feature_matrix import (
    calculate_feature_matrix_spark,
)
from featuretools.computational_backends.spark.spark_feature_set_calculator import (
    SparkFeatureSetCalculator,
)

__all__ = ["calculate_feature_matrix_spark", "SparkFeatureSetCalculator"]

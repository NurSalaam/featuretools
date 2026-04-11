"""
Native PySpark backend for Featuretools EntitySets.

This subpackage provides :class:`SparkEntitySet`, a drop-in subclass of
:class:`featuretools.EntitySet` whose internal dataframes are
``pyspark.sql.DataFrame`` objects instead of pandas DataFrames. When a
``SparkEntitySet`` is passed to :func:`ft.dfs` or
:func:`ft.calculate_feature_matrix`, the computation is dispatched to the
native Spark execution engine in
:mod:`featuretools.computational_backends.spark`.

Unlike the historical Dask/Koalas support removed in PR #2705, this path does
not use a pandas-API-on-Spark compatibility shim. It emits real Spark
DataFrame operations (``groupBy``, ``agg``, ``join``, window functions) and
falls back to vectorized ``pandas_udf`` only for primitives that lack a
Spark-SQL equivalent.
"""
from featuretools.entityset.spark.spark_entityset import SparkEntitySet
from featuretools.entityset.spark.spark_schema import SparkLogicalSchema

__all__ = ["SparkEntitySet", "SparkLogicalSchema"]

"""
Top-level orchestration for native Spark DFS.

Analog of
:func:`featuretools.computational_backends.calculate_feature_matrix.calculate_feature_matrix`
but operates over a :class:`SparkEntitySet`. See that module for the
pandas-path reference.

The entry point :func:`calculate_feature_matrix_spark` validates inputs,
builds a ``FeatureSet``, instantiates :class:`SparkFeatureSetCalculator`,
runs it, and returns either a pandas DataFrame (``return_spark=False``,
default, for API parity) or the lazy Spark DataFrame (``return_spark=True``).
"""
from __future__ import annotations

import logging
import warnings
from typing import Iterable, List, Optional

from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.spark.spark_cutoff_time import (
    normalize_cutoff_time,
)
from featuretools.computational_backends.spark.spark_feature_set_calculator import (
    SparkFeatureSetCalculator,
)
from featuretools.feature_base import AggregationFeature, FeatureBase

logger = logging.getLogger("featuretools.computational_backends.spark")


def calculate_feature_matrix_spark(
    features: List[FeatureBase],
    entityset,
    cutoff_time=None,
    instance_ids: Optional[Iterable] = None,
    training_window=None,
    approximate=None,
    save_progress=None,
    verbose: bool = False,
    chunk_size=None,
    n_jobs: int = 1,
    dask_kwargs=None,
    progress_callback=None,
    include_cutoff_time: bool = True,
    return_spark: bool = False,
):
    """Calculate a feature matrix for a :class:`SparkEntitySet`.

    Accepts the same signature as the pandas ``calculate_feature_matrix``
    so the top-level dispatch can forward arguments wholesale.
    ``n_jobs``, ``chunk_size``, ``dask_kwargs``, ``save_progress``, and
    ``approximate`` are not meaningful on the Spark path — they are
    accepted for signature parity but warned about if non-default.

    Parameters
    ----------
    return_spark : bool
        If True, return the underlying lazy ``pyspark.sql.DataFrame``
        without ``.toPandas()``. Useful for downstream Spark work.
    """
    from featuretools.entityset.spark.spark_entityset import SparkEntitySet

    if not isinstance(features, list) or not features:
        raise ValueError("features must be a non-empty list of FeatureBase")
    if not all(isinstance(f, FeatureBase) for f in features):
        raise TypeError("features must all be FeatureBase instances")
    if not isinstance(entityset, SparkEntitySet):
        raise TypeError(
            "calculate_feature_matrix_spark requires a SparkEntitySet; "
            f"got {type(entityset).__name__}"
        )

    # Warn about parameters that have no effect on the Spark path.
    if n_jobs != 1:
        warnings.warn(
            "n_jobs is ignored for SparkEntitySet; Spark manages executor "
            "parallelism via the SparkSession configuration."
        )
    if chunk_size is not None:
        warnings.warn(
            "chunk_size is ignored for SparkEntitySet; Spark partitions "
            "are managed by the underlying DataFrame."
        )
    if dask_kwargs:
        warnings.warn("dask_kwargs is ignored for SparkEntitySet.")
    if save_progress is not None:
        warnings.warn(
            "save_progress is ignored for SparkEntitySet; use "
            "spark_df.write.parquet(...) on the returned lazy frame."
        )
    if approximate is not None:
        raise NotImplementedError(
            "approximate features are not supported on the Spark backend"
        )

    scalar_cutoff, cutoff_df, ct_instance_ids = normalize_cutoff_time(
        cutoff_time, entityset, features[0].dataframe_name
    )
    if cutoff_df is not None:
        # First cut: per-instance cutoff_time uses max(time) as the scalar
        # and filters to the given instance_ids. This is an intentional
        # simplification documented in the plan; full non-equi join
        # support is a v2 enhancement.
        warnings.warn(
            "Per-instance cutoff_time on the Spark backend is currently "
            "implemented as max(cutoff_time.time) + filter on instance_ids. "
            "Per-row time semantics will land in a follow-up."
        )

    if instance_ids is None:
        instance_ids = ct_instance_ids

    feature_set = FeatureSet(features)
    calculator = SparkFeatureSetCalculator(
        entityset=entityset,
        feature_set=feature_set,
        time_last=scalar_cutoff,
        training_window=training_window,
    )

    result = calculator.run(
        instance_ids=instance_ids,
        progress_callback=progress_callback,
        include_cutoff_time=include_cutoff_time,
    )

    if return_spark:
        return result

    # Materialize to pandas for API parity with the pandas backend.
    pdf = result.toPandas()
    target_index = entityset.schemas[feature_set.target_df_name].index
    if target_index in pdf.columns:
        pdf = pdf.set_index(target_index)
    return pdf

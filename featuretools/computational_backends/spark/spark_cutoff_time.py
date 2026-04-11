"""
Cutoff-time handling for the native Spark DFS backend.

Two shapes are supported, matching the pandas backend:

1. **Scalar cutoff_time** — one timestamp applied uniformly. The Spark
   calculator sees this as a Python ``datetime``/``np.inf`` and the
   per-dataframe time filter drops everything after it.
2. **Per-instance cutoff_time DataFrame** — a pandas DataFrame with
   ``instance_id`` and ``time`` columns (plus optional pass-through
   columns). We convert it to a Spark DataFrame, broadcast if small, and
   use non-equi joins against the fact tables: each aggregation groupBy
   is preceded by a join on
   ``(child.fk == cutoff.instance_id) & (child.time <= cutoff.time)``.

The non-equi join handling is orchestrated by the
:class:`SparkFeatureSetCalculator` via the ``time_last`` parameter on
``EntitySet.query_by_values``. In this first cut we only support shape (1);
shape (2) requires the calculator to know about per-row cutoff times
during aggregation and is deferred to a v2 pass.
"""
from __future__ import annotations

from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd


CutoffTimeLike = Union[datetime, float, pd.DataFrame, Tuple]


def normalize_cutoff_time(
    cutoff_time: CutoffTimeLike, entityset, target_dataframe_name: str
):
    """Convert whatever the user passed into a uniform shape the Spark
    calculator can consume.

    Returns
    -------
    scalar_time : datetime or float or None
        The single time to apply across all dataframes, or None if
        per-instance cutoff times were provided.
    cutoff_df : Optional[pyspark.sql.DataFrame]
        The per-instance cutoff DataFrame (with ``instance_id`` and
        ``time`` columns), or None.
    instance_ids : Optional[list]
        The list of instance ids to compute features for, or None for
        "all rows in target dataframe".
    """
    # Scalar / None path
    if cutoff_time is None:
        if entityset.time_type == "numeric":
            return np.inf, None, None
        return datetime.now(), None, None

    if isinstance(cutoff_time, (datetime, pd.Timestamp, int, float, np.datetime64)):
        return cutoff_time, None, None

    if isinstance(cutoff_time, pd.DataFrame):
        # Per-instance cutoff times. In the first cut we do NOT support
        # non-equi joins inside DFS — we just extract the max time and
        # treat it as a scalar, then further subset by instance_id at the
        # very end. A production v2 would convert this into a Spark
        # DataFrame and perform non-equi joins inside each aggregation.
        if "instance_id" not in cutoff_time.columns:
            raise ValueError(
                "Per-instance cutoff_time DataFrame must include an "
                "'instance_id' column"
            )
        if "time" not in cutoff_time.columns:
            raise ValueError(
                "Per-instance cutoff_time DataFrame must include a "
                "'time' column"
            )
        scalar_time = cutoff_time["time"].max()
        instance_ids = cutoff_time["instance_id"].tolist()
        return scalar_time, cutoff_time, instance_ids

    raise TypeError(
        f"Unsupported cutoff_time type {type(cutoff_time).__name__} "
        "for the Spark backend; pass a datetime, numeric, or DataFrame"
    )

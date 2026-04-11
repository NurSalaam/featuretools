"""
SparkEntitySet — the Spark-native analog of :class:`featuretools.EntitySet`.

Stores ``pyspark.sql.DataFrame`` values in ``dataframe_dict`` instead of
pandas DataFrames, and carries a parallel ``schemas`` dict of
:class:`SparkLogicalSchema` objects instead of leaning on woodwork's ``.ww``
accessor.

Subclasses :class:`EntitySet` so the relationship graph, feature-base
metadata, ``find_backward_paths``, and most plumbing work unchanged. Only the
data-touching methods (``add_dataframe``, ``query_by_values``, ``_handle_time``)
are overridden.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

from featuretools.entityset.entityset import EntitySet, LTI_COLUMN_NAME
from featuretools.entityset.relationship import Relationship
from featuretools.entityset.spark.spark_schema import SparkLogicalSchema
from featuretools.utils.wrangle import _check_timedelta

logger = logging.getLogger("featuretools.entityset.spark")


class SparkEntitySet(EntitySet):
    """A Featuretools EntitySet whose internal tables are Spark DataFrames.

    Parameters
    ----------
    id : str
        Unique identifier.
    dataframes : dict, optional
        Mapping of name →
        ``(spark_dataframe, index, time_index, logical_types, foreign_keys)``
        where ``logical_types`` is ``dict[str, str]`` (see
        :data:`SparkLogicalSchema.KNOWN_LOGICAL_TYPES`) and ``foreign_keys``
        is a set of column names. The tuple shape is intentionally simpler
        than the pandas EntitySet's 6-tuple because the Spark path does not
        support ``make_index`` or ``semantic_tags`` beyond FK/PK/time_index.
    relationships : list, optional
        List of ``(parent_name, parent_col, child_name, child_col)`` tuples.

    Notes
    -----
    - ``pyspark`` is imported lazily so this class can be imported without a
      Spark installation; only instantiation triggers the import.
    - We do **not** subclass the woodwork init codepath used by the parent
      class. Instead we bypass the parent ``__init__`` entirely and
      reimplement its minimal bookkeeping (``id``, ``dataframe_dict``,
      ``relationships``, ``time_type``). This avoids the woodwork ``.ww``
      access in ``EntitySet.__init__`` at line 79.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        dataframes: Optional[Dict[str, Tuple]] = None,
        relationships: Optional[List[Tuple[str, str, str, str]]] = None,
    ):
        # Bypass EntitySet.__init__ because it calls df.ww.schema on every
        # dataframe in the input dict, which doesn't exist on
        # pyspark.sql.DataFrame. We reimplement the minimal bookkeeping here.
        self.id = id
        self.dataframe_dict: Dict[str, "pyspark.sql.DataFrame"] = {}  # noqa: F821
        self.schemas: Dict[str, SparkLogicalSchema] = {}
        self.relationships: List[Relationship] = []
        self.time_type = None
        self._data_description = None

        dataframes = dataframes or {}
        relationships = relationships or []

        for df_name, spec in dataframes.items():
            spark_df = spec[0]
            index_col = spec[1] if len(spec) > 1 else None
            time_index = spec[2] if len(spec) > 2 else None
            logical_types = spec[3] if len(spec) > 3 else None
            foreign_keys = spec[4] if len(spec) > 4 else None
            self.add_dataframe(
                dataframe_name=df_name,
                dataframe=spark_df,
                index=index_col,
                time_index=time_index,
                logical_types=logical_types,
                foreign_keys=foreign_keys,
            )

        for rel in relationships:
            parent_df, parent_col, child_df, child_col = rel
            self.add_relationship(parent_df, parent_col, child_df, child_col)

        # Register in the module-level ES reference cache, mirroring
        # EntitySet.__init__ at line 114 so FeatureBase lookups resolve.
        from featuretools.feature_base.feature_base import _ES_REF

        _ES_REF[self.id] = self

    # ------------------------------------------------------------------
    # Dataframe registration
    # ------------------------------------------------------------------
    def add_dataframe(
        self,
        dataframe_name: str,
        dataframe,
        index: Optional[str] = None,
        time_index: Optional[str] = None,
        logical_types: Optional[Dict[str, str]] = None,
        foreign_keys: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        """Register a Spark DataFrame on the EntitySet.

        Unlike :meth:`EntitySet.add_dataframe` this accepts a Spark DataFrame
        and explicit type/index kwargs rather than woodwork-typed inputs.
        """
        from pyspark.sql import DataFrame as SparkDataFrame  # local import

        if not isinstance(dataframe, SparkDataFrame):
            raise TypeError(
                "SparkEntitySet.add_dataframe requires a pyspark.sql.DataFrame; "
                f"got {type(dataframe).__name__}. Use EntitySet for pandas."
            )

        if dataframe_name in self.dataframe_dict:
            raise ValueError(
                f"Dataframe '{dataframe_name}' already exists in SparkEntitySet"
            )

        if index is None:
            raise ValueError(
                "SparkEntitySet.add_dataframe requires an explicit `index` column. "
                "Index auto-creation is not supported on the Spark path."
            )

        if index not in dataframe.columns:
            raise ValueError(
                f"Index column '{index}' not found in dataframe '{dataframe_name}'"
            )
        if time_index is not None and time_index not in dataframe.columns:
            raise ValueError(
                f"time_index '{time_index}' not found in dataframe '{dataframe_name}'"
            )

        inferred_types = _infer_logical_types(dataframe)
        if logical_types:
            inferred_types.update(logical_types)

        schema = SparkLogicalSchema(
            name=dataframe_name,
            index=index,
            time_index=time_index,
            logical_types=inferred_types,
            foreign_keys=set(foreign_keys or ()),
        )

        self.dataframe_dict[dataframe_name] = dataframe
        self.schemas[dataframe_name] = schema

        # Track time_type for the whole EntitySet (matches pandas EntitySet
        # behavior, used by FeatureSet for cutoff-time validation).
        if time_index is not None and self.time_type is None:
            self.time_type = _guess_time_type(dataframe, time_index)

        self._data_description = None

    # ------------------------------------------------------------------
    # Relationship registration
    # ------------------------------------------------------------------
    def add_relationship(
        self,
        parent_dataframe_name=None,
        parent_column_name=None,
        child_dataframe_name=None,
        child_column_name=None,
        relationship=None,
    ):
        """Register a parent→child relationship.

        We cannot call ``Relationship(self, ...)`` directly because its
        ``__init__`` dereferences ``parent_dataframe.ww.index`` (see
        relationship.py:33), which doesn't exist on Spark DataFrames.
        Instead we construct the object via ``object.__new__`` and set the
        four name attributes directly — they're the only ones the DFS graph
        traversal code touches.
        """
        if relationship is not None:
            parent_dataframe_name = relationship._parent_dataframe_name
            parent_column_name = relationship._parent_column_name
            child_dataframe_name = relationship._child_dataframe_name
            child_column_name = relationship._child_column_name

        # Validate that referenced dataframes/columns exist on the
        # SparkEntitySet before registering the relationship. Mirrors the
        # sanity checks that Relationship.__init__ normally performs.
        parent_df = self.dataframe_dict.get(parent_dataframe_name)
        child_df = self.dataframe_dict.get(child_dataframe_name)
        if parent_df is None or child_df is None:
            raise KeyError(
                f"Both '{parent_dataframe_name}' and '{child_dataframe_name}' "
                "must be added to the SparkEntitySet before adding a relationship"
            )
        if parent_column_name not in parent_df.columns:
            raise ValueError(
                f"Parent column '{parent_column_name}' not in dataframe "
                f"'{parent_dataframe_name}'"
            )
        if child_column_name not in child_df.columns:
            raise ValueError(
                f"Child column '{child_column_name}' not in dataframe "
                f"'{child_dataframe_name}'"
            )

        # Enforce that parent_column is the declared index on the parent,
        # matching the pandas Relationship invariant.
        parent_schema = self.schemas[parent_dataframe_name]
        if parent_column_name != parent_schema.index:
            raise ValueError(
                f"Parent column '{parent_column_name}' is not the index of "
                f"dataframe '{parent_dataframe_name}' "
                f"(index is '{parent_schema.index}')"
            )

        if relationship is None:
            relationship = object.__new__(Relationship)
            relationship.entityset = self
            relationship._parent_dataframe_name = parent_dataframe_name
            relationship._child_dataframe_name = child_dataframe_name
            relationship._parent_column_name = parent_column_name
            relationship._child_column_name = child_column_name

        if relationship in self.relationships:
            warnings.warn(f"Relationship {relationship} already exists", UserWarning)
            return

        self.relationships.append(relationship)
        self.schemas[child_dataframe_name].mark_foreign_key(child_column_name)
        self._data_description = None

    # ------------------------------------------------------------------
    # Query — the engine's main entry point into the EntitySet
    # ------------------------------------------------------------------
    def query_by_values(
        self,
        dataframe_name,
        instance_vals,
        column_name=None,
        columns=None,
        time_last=None,
        training_window=None,
        include_cutoff_time=True,
    ):
        """Lazy Spark equivalent of :meth:`EntitySet.query_by_values`.

        Returns an unmaterialized ``pyspark.sql.DataFrame``. No ``.collect()``
        or ``.toPandas()`` happens here — the final materialization is driven
        by :meth:`SparkFeatureSetCalculator.run` at the very end of DFS.
        """
        from pyspark.sql import functions as F  # local import

        schema = self.schemas[dataframe_name]
        df = self.dataframe_dict[dataframe_name]

        if column_name is None:
            column_name = schema.index

        # Filter by instance_vals if provided.
        if instance_vals is not None:
            vals = _normalize_instance_vals(instance_vals)
            if len(vals) == 0:
                # Match pandas semantics: empty filter returns empty dataframe.
                df = df.limit(0)
            else:
                df = df.filter(F.col(column_name).isin(vals))

        df = self._handle_time(
            dataframe_name=dataframe_name,
            df=df,
            time_last=time_last,
            training_window=training_window,
            include_cutoff_time=include_cutoff_time,
        )

        if columns is not None:
            df = df.select(*[F.col(c) for c in columns])

        return df

    # ------------------------------------------------------------------
    # Time filtering
    # ------------------------------------------------------------------
    def _handle_time(
        self,
        dataframe_name,
        df,
        time_last=None,
        training_window=None,
        include_cutoff_time=True,
    ):
        """Mirror of :meth:`EntitySet._handle_time` using Spark filters."""
        from pyspark.sql import functions as F

        schema = self.schemas[dataframe_name]

        if schema.time_index and time_last is not None:
            time_col = F.col(schema.time_index)
            time_lit = F.lit(time_last)
            if include_cutoff_time:
                df = df.filter(time_col <= time_lit)
            else:
                df = df.filter(time_col < time_lit)

            if training_window is not None:
                training_window = _check_timedelta(training_window)
                window_start = time_last - training_window
                window_start_lit = F.lit(window_start)
                if include_cutoff_time:
                    mask = F.col(schema.time_index) > window_start_lit
                else:
                    mask = F.col(schema.time_index) >= window_start_lit

                lti_col = schema.last_time_index_col
                if lti_col is not None:
                    if include_cutoff_time:
                        lti_mask = F.col(lti_col) > window_start_lit
                    else:
                        lti_mask = F.col(lti_col) >= window_start_lit
                    mask = mask | lti_mask
                else:
                    warnings.warn(
                        f"Using training_window but last_time_index is not "
                        f"set for dataframe {dataframe_name}"
                    )

                df = df.filter(mask)

        # Secondary time indexes: mask out their dependent columns after
        # time_last. Mirrors entityset.py:1450.
        for sec_idx, cols in schema.secondary_time_indexes.items():
            if time_last is not None:
                mask = F.col(sec_idx) >= F.lit(time_last)
                for c in cols:
                    df = df.withColumn(
                        c, F.when(mask, F.lit(None)).otherwise(F.col(c))
                    )

        return df

    # ------------------------------------------------------------------
    # No-op / disabled methods on the Spark path
    # ------------------------------------------------------------------
    def _check_time_indexes(self):
        """Verify all dataframes with a time_index agree on numeric vs
        datetime. On the Spark path we rely on :meth:`add_dataframe` having
        set ``time_type`` at registration time, so this is a no-op."""
        return

    def add_last_time_indexes(self, *args, **kwargs):
        raise NotImplementedError(
            "add_last_time_indexes is not supported on SparkEntitySet. "
            "Pre-compute the last-time-index column and pass it via the "
            "`logical_types` dict when calling add_dataframe."
        )

    def to_pickle(self, *args, **kwargs):
        raise NotImplementedError(
            "SparkEntitySet serialization is not supported. Persist the "
            "underlying Spark DataFrames via spark_df.write.parquet(...) "
            "and re-construct the EntitySet on load."
        )

    to_csv = to_pickle
    to_parquet = to_pickle

    def __getitem__(self, dataframe_name):
        if dataframe_name in self.dataframe_dict:
            return self.dataframe_dict[dataframe_name]
        raise KeyError(
            f"DataFrame {dataframe_name} does not exist in "
            f"SparkEntitySet({self.id!r})"
        )

    def __repr__(self):
        return (
            f"SparkEntitySet(id={self.id!r}, "
            f"dataframes={list(self.dataframe_dict.keys())}, "
            f"relationships={len(self.relationships)})"
        )

    # ------------------------------------------------------------------
    # EntitySet parent class methods that would touch woodwork — override
    # to use SparkLogicalSchema instead.
    # ------------------------------------------------------------------
    def reset_data_description(self):
        self._data_description = None

    def _add_references_to_metadata(self, *args, **kwargs):
        # Pandas EntitySet uses this to register ww accessors on copies. No-op
        # on Spark.
        return


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _infer_logical_types(spark_df) -> Dict[str, str]:
    """Infer a ``SparkLogicalSchema``-compatible logical-type map from a
    Spark DataFrame's schema. Falls back to ``"Unknown"`` for unrecognized
    Spark types."""
    try:
        from pyspark.sql import types as T
    except ImportError:
        return {}

    mapping = {
        T.ByteType: "Integer",
        T.ShortType: "Integer",
        T.IntegerType: "Integer",
        T.LongType: "Integer",
        T.FloatType: "Double",
        T.DoubleType: "Double",
        T.DecimalType: "Double",
        T.BooleanType: "Boolean",
        T.TimestampType: "Datetime",
        T.TimestampNTZType: "Datetime",
        T.DateType: "Datetime",
        T.DayTimeIntervalType: "Timedelta",
        T.StringType: "NaturalLanguage",
    }
    out: Dict[str, str] = {}
    for field in spark_df.schema.fields:
        dtype = type(field.dataType)
        out[field.name] = mapping.get(dtype, "Unknown")
    return out


def _guess_time_type(spark_df, time_index: str) -> str:
    try:
        from pyspark.sql import types as T
    except ImportError:
        return "datetime"

    dtype = spark_df.schema[time_index].dataType
    if isinstance(
        dtype, (T.TimestampType, T.TimestampNTZType, T.DateType)
    ):
        return "datetime"
    if isinstance(
        dtype,
        (
            T.ByteType,
            T.ShortType,
            T.IntegerType,
            T.LongType,
            T.FloatType,
            T.DoubleType,
            T.DecimalType,
        ),
    ):
        return "numeric"
    return "datetime"


def _normalize_instance_vals(instance_vals) -> List:
    """Coerce instance_vals into a plain Python list that Spark's ``isin``
    accepts. Handles pd.Series, np.ndarray, pd.Categorical, tuples, scalars.
    """
    import pandas as pd
    import numpy as np

    if isinstance(instance_vals, pd.Series):
        return instance_vals.dropna().unique().tolist()
    if isinstance(instance_vals, pd.Index):
        return instance_vals.tolist()
    if isinstance(instance_vals, np.ndarray):
        return instance_vals.tolist()
    if isinstance(instance_vals, (list, tuple, set)):
        return list(instance_vals)
    return [instance_vals]

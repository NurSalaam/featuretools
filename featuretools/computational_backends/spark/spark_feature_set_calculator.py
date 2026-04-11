"""
Native-Spark analog of
:class:`featuretools.computational_backends.feature_set_calculator.FeatureSetCalculator`.

Structural contract (kept intentionally close to the pandas calculator so
that code review can cross-reference line for line):

- ``run(instance_ids)``                 ↔ feature_set_calculator.py:68
- ``_calculate_features_for_dataframe`` ↔ feature_set_calculator.py:163
- ``_calculate_features``               ↔ feature_set_calculator.py:360
- ``_feature_type_handler``             ↔ feature_set_calculator.py:456
- ``_calculate_identity_features``      ↔ feature_set_calculator.py:470
- ``_calculate_transform_features``     ↔ feature_set_calculator.py:480
- ``_calculate_groupby_features``       ↔ feature_set_calculator.py:526
- ``_calculate_direct_features``        ↔ feature_set_calculator.py:594
- ``_calculate_agg_features``           ↔ feature_set_calculator.py:652

But every operation produces a **lazy** ``pyspark.sql.DataFrame`` instead of
a materialized pandas frame. The final ``.toPandas()`` happens at the very
end of ``run()`` (or is skipped if the caller passes ``return_spark=True``).
Catalyst fuses the chained plan into one job.

No chunking, no ``parallel_calculate_chunks``. Spark owns parallelism.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from featuretools.computational_backends.spark.spark_primitive_dispatch import (
    aggregation_feature_expr,
    groupby_transform_feature_expr,
    transform_feature_expr,
)
from featuretools.computational_backends.spark.spark_utils import (
    default_to_spark_literal,
    should_broadcast,
)
from featuretools.entityset.relationship import RelationshipPath
from featuretools.exceptions import UnknownFeature
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from featuretools.utils import Trie

logger = logging.getLogger("featuretools.computational_backends.spark")


class SparkFeatureSetCalculator:
    """Compute a feature matrix against a :class:`SparkEntitySet`.

    Unlike the pandas calculator this class only produces lazy Spark plans —
    no action (``.collect()``, ``.count()``, ``.toPandas()``) is performed
    inside the recursive traversal. The caller (usually
    :func:`calculate_feature_matrix_spark`) decides when to materialize.
    """

    def __init__(
        self,
        entityset,
        feature_set,
        time_last=None,
        training_window=None,
        precalculated_features=None,
    ):
        self.entityset = entityset
        self.feature_set = feature_set
        self.training_window = training_window
        self.time_last = time_last if time_last is not None else datetime.now()

        if precalculated_features is None:
            precalculated_features = Trie(path_constructor=RelationshipPath)
        self.precalculated_features = precalculated_features

        self.num_features = sum(
            len(features1) + len(features2)
            for _, (_, features1, features2) in self.feature_set.feature_trie
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(
        self,
        instance_ids=None,
        progress_callback=None,
        include_cutoff_time: bool = True,
    ):
        """Walk the feature trie and build a lazy Spark plan.

        Parameters
        ----------
        instance_ids : Optional[Iterable]
            If provided, restricts the target dataframe to rows whose index
            matches. Otherwise the full target dataframe is used.
        progress_callback : Optional[callable]
            Unused on the Spark path (Spark handles its own progress via
            the SparkUI). Accepted for API parity with the pandas
            calculator.
        include_cutoff_time : bool
            Whether data exactly at ``time_last`` is included.

        Returns
        -------
        pyspark.sql.DataFrame
            Lazy Spark DataFrame containing one column per target feature
            plus the target index column.
        """
        if progress_callback is None:
            def progress_callback(*args):
                return

        feature_trie = self.feature_set.feature_trie
        df_trie: Trie = Trie(path_constructor=RelationshipPath)
        full_df_trie: Trie = Trie(path_constructor=RelationshipPath)

        target_name = self.feature_set.target_df_name
        target_schema = self.entityset.schemas[target_name]

        self._calculate_features_for_dataframe(
            dataframe_name=target_name,
            feature_trie=feature_trie,
            df_trie=df_trie,
            full_dataframe_trie=full_df_trie,
            precalculated_trie=self.precalculated_features,
            filter_column=target_schema.index,
            filter_values=instance_ids,
            parent_data=None,
            progress_callback=progress_callback,
            include_cutoff_time=include_cutoff_time,
        )

        result_df = df_trie.value

        # Select target feature columns (plus the target index so callers
        # can join the result back).
        from pyspark.sql import functions as F

        target_cols: List[str] = [target_schema.index]
        seen: Set[str] = {target_schema.index}
        for feat in self.feature_set.target_features:
            for name in feat.get_feature_names():
                if name not in seen:
                    target_cols.append(name)
                    seen.add(name)

        # Some columns may not exist because the feature computation produced
        # nulls or was dropped downstream; fall back to F.lit(None).
        select_cols = []
        existing = set(result_df.columns)
        for col in target_cols:
            if col in existing:
                select_cols.append(F.col(col))
            else:
                select_cols.append(F.lit(None).alias(col))
        return result_df.select(*select_cols)

    # ------------------------------------------------------------------
    # Recursive trie traversal
    # ------------------------------------------------------------------
    def _calculate_features_for_dataframe(
        self,
        dataframe_name: str,
        feature_trie,
        df_trie: Trie,
        full_dataframe_trie: Trie,
        precalculated_trie: Trie,
        filter_column: str,
        filter_values,
        parent_data: Optional[Tuple] = None,
        progress_callback=None,
        include_cutoff_time: bool = True,
    ) -> None:
        """Spark analog of
        ``FeatureSetCalculator._calculate_features_for_dataframe``.

        ``filter_values`` can be:
        - ``None`` — no filter (use the full child dataframe), used when a
          descendant feature is ``uses_full_dataframe`` or when we're at
          the root and no ``instance_ids`` were provided.
        - a Python sequence — treated as an IN-filter via ``.isin()``.
        - a ``pyspark.sql.DataFrame`` with a single column — used for
          semi-join-based subsetting when the parent-side key set is
          computed lazily.
        """
        from pyspark.sql import DataFrame as SparkDataFrame, functions as F

        (
            need_full_dataframe,
            full_df_features,
            not_full_df_features,
        ) = feature_trie.value
        all_features = full_df_features | not_full_df_features
        columns = self._necessary_columns(dataframe_name, all_features)

        # Step 1: pull a filtered lazy plan for this dataframe.
        if need_full_dataframe:
            query_col = None
            query_vals = None
        else:
            query_col = filter_column
            query_vals = filter_values

        if isinstance(query_vals, SparkDataFrame):
            # Caller handed us a lazy key set. Bypass query_by_values's
            # instance-vals pathway (which expects a Python sequence) and
            # do a semi-join instead.
            full_df = self.entityset.query_by_values(
                dataframe_name=dataframe_name,
                instance_vals=None,
                column_name=query_col,
                columns=columns,
                time_last=self.time_last,
                training_window=self.training_window,
                include_cutoff_time=include_cutoff_time,
            )
            key_col = query_vals.columns[0]
            df = full_df.join(
                query_vals.select(F.col(key_col).alias("_ft_filter_key")).dropDuplicates(),
                full_df[query_col] == F.col("_ft_filter_key"),
                "left_semi",
            )
        else:
            df = self.entityset.query_by_values(
                dataframe_name=dataframe_name,
                instance_vals=query_vals,
                column_name=query_col,
                columns=columns,
                time_last=self.time_last,
                training_window=self.training_window,
                include_cutoff_time=include_cutoff_time,
            )

        progress_callback(0)

        # Step 2: add ancestor-linkage columns so deep aggregations can join
        # all the way back. Mirror of feature_set_calculator.py:248.
        new_ancestor_rel_columns: List[str] = []
        if parent_data is not None:
            parent_relationship, ancestor_rel_columns, parent_df = parent_data
            if ancestor_rel_columns:
                df, new_ancestor_rel_columns = (
                    self._add_ancestor_relationship_columns(
                        df,
                        parent_df,
                        ancestor_rel_columns,
                        parent_relationship,
                    )
                )
            new_ancestor_rel_columns.append(
                parent_relationship._child_column_name
            )

        # Step 3: recurse on children. For each edge in the trie, the child
        # frame is filtered to the set of keys implied by the currently
        # computed (lazy) parent frame.
        for edge, sub_trie in feature_trie.children():
            is_forward, relationship = edge
            if is_forward:
                sub_dataframe_name = relationship._parent_dataframe_name
                sub_filter_column = relationship._parent_column_name
                # Forward: child's FK values act as the filter set on the
                # parent. We pass a lazy Spark DataFrame with one column.
                sub_filter_values = df.select(
                    F.col(relationship._child_column_name).alias("key")
                )
                sub_parent_data = None
            else:
                sub_dataframe_name = relationship._child_dataframe_name
                sub_filter_column = relationship._child_column_name
                sub_filter_values = df.select(
                    F.col(relationship._parent_column_name).alias("key")
                )
                sub_parent_data = (
                    relationship,
                    new_ancestor_rel_columns,
                    df,
                )

            sub_df_trie = df_trie.get_node([edge])
            sub_full_trie = full_dataframe_trie.get_node([edge])
            sub_precalc_trie = precalculated_trie.get_node([edge])
            self._calculate_features_for_dataframe(
                dataframe_name=sub_dataframe_name,
                feature_trie=sub_trie,
                df_trie=sub_df_trie,
                full_dataframe_trie=sub_full_trie,
                precalculated_trie=sub_precalc_trie,
                filter_column=sub_filter_column,
                filter_values=sub_filter_values,
                parent_data=sub_parent_data,
                progress_callback=progress_callback,
                include_cutoff_time=include_cutoff_time,
            )

        # Step 4: calculate this dataframe's own features.
        if need_full_dataframe:
            df = self._calculate_features(
                df, full_dataframe_trie, full_df_features, progress_callback
            )
            full_dataframe_trie.value = df
            # Narrow to filter_values for the non-full features, matching
            # pandas behavior at feature_set_calculator.py:346.
            if filter_values is not None:
                if isinstance(filter_values, SparkDataFrame):
                    key_col = filter_values.columns[0]
                    df = df.join(
                        filter_values.select(
                            F.col(key_col).alias("_ft_filter_key")
                        ).dropDuplicates(),
                        df[filter_column] == F.col("_ft_filter_key"),
                        "left_semi",
                    )
                else:
                    from featuretools.entityset.spark.spark_entityset import (
                        _normalize_instance_vals,
                    )
                    vals = _normalize_instance_vals(filter_values)
                    if len(vals):
                        df = df.filter(F.col(filter_column).isin(vals))

        df = self._calculate_features(
            df, df_trie, not_full_df_features, progress_callback
        )

        # Step 5: stash this node's lazy plan for parent recursion.
        df_trie.value = df

    # ------------------------------------------------------------------
    # Feature-type dispatch
    # ------------------------------------------------------------------
    def _calculate_features(self, df, df_trie, feature_names, progress_callback):
        feature_groups = self.feature_set.group_features(feature_names)
        for group in feature_groups:
            representative = group[0]
            handler = self._feature_type_handler(representative)
            df = handler(group, df, df_trie, progress_callback)
        return df

    def _feature_type_handler(self, f):
        if type(f) == TransformFeature:
            return self._calculate_transform_features
        if type(f) == GroupByTransformFeature:
            return self._calculate_groupby_features
        if type(f) == DirectFeature:
            return self._calculate_direct_features
        if type(f) == AggregationFeature:
            return self._calculate_agg_features
        if type(f) == IdentityFeature:
            return self._calculate_identity_features
        raise UnknownFeature(f"{f.__class__} feature unknown")

    # ------------------------------------------------------------------
    # Identity — passthrough
    # ------------------------------------------------------------------
    def _calculate_identity_features(
        self, features, df, _df_trie, progress_callback
    ):
        for f in features:
            assert f.get_name() in df.columns, (
                f'Column "{f.get_name()}" missing from Spark dataframe'
            )
        progress_callback(len(features) / float(max(self.num_features, 1)))
        return df

    # ------------------------------------------------------------------
    # Transform — column expressions
    # ------------------------------------------------------------------
    def _calculate_transform_features(
        self, features, df, _df_trie, progress_callback
    ):
        from pyspark.sql import functions as F

        for f in features:
            input_cols = [F.col(bf.get_name()) for bf in f.base_features]
            expr = transform_feature_expr(f, input_cols)
            if f.number_output_features > 1:
                # Multi-output: dispatcher returns a list of Columns.
                if not isinstance(expr, (list, tuple)):
                    raise ValueError(
                        f"Multi-output primitive {type(f.primitive).__name__} "
                        "must return a list of Columns from its Spark dispatch"
                    )
                for name, col in zip(f.get_feature_names(), expr):
                    df = df.withColumn(name, col)
            else:
                df = df.withColumn(f.get_name(), expr)
            progress_callback(1 / float(max(self.num_features, 1)))
        return df

    # ------------------------------------------------------------------
    # GroupByTransform — window functions
    # ------------------------------------------------------------------
    def _calculate_groupby_features(
        self, features, df, _df_trie, progress_callback
    ):
        from pyspark.sql import functions as F

        if not features:
            return df

        groupby_name = features[0].groupby.get_name()
        # Use the child dataframe's time_index column (if any) as the window
        # order. We can't easily look it up through the pandas entityset
        # because we might be operating on a filtered projection; the
        # schema is keyed on the *original* dataframe name, which we get
        # from the feature's dataframe attribute.
        order_col: Optional[str] = None
        try:
            schema = self.entityset.schemas.get(features[0].dataframe_name)
            if schema is not None and schema.time_index in df.columns:
                order_col = schema.time_index
        except AttributeError:
            order_col = None

        for f in features:
            base_col = f.base_features[0].get_name()
            try:
                expr = groupby_transform_feature_expr(
                    f, base_col, groupby_name, order_col
                )
            except NotImplementedError:
                # Fall through to pandas_udf grouped-map in a future
                # iteration; for now surface a clear error.
                raise
            df = df.withColumn(f.get_name(), expr)
            progress_callback(1 / float(max(self.num_features, 1)))
        return df

    # ------------------------------------------------------------------
    # DirectFeature — join parent columns into child
    # ------------------------------------------------------------------
    def _calculate_direct_features(
        self, features, child_df, df_trie, progress_callback
    ):
        from pyspark.sql import functions as F

        path = features[0].relationship_path
        assert len(path) == 1, "DirectFeature should have single-hop relationship path"

        _is_forward, relationship = path[0]
        parent_df = df_trie.get_node([path[0]]).value
        merge_col = relationship._child_column_name
        parent_key = relationship._parent_column_name

        col_map: Dict[str, str] = {parent_key: merge_col}
        default_exprs: Dict[str, "pyspark.sql.Column"] = {}  # noqa: F821
        for f in features:
            base_name = f.base_features[0].get_feature_names()[0]
            for name, bn in zip(f.get_feature_names(), f.base_features[0].get_feature_names()):
                if name in child_df.columns:
                    continue
                col_map[bn] = name
            if f.default_value is not None:
                import pandas as pd
                if not (isinstance(f.default_value, float) and pd.isna(f.default_value)):
                    default_exprs[f.get_name()] = default_to_spark_literal(
                        f.default_value
                    )

        # Project parent to only the columns we'll carry across, and rename
        # to the child-side feature names.
        project_cols = []
        for src, dst in col_map.items():
            if src not in parent_df.columns:
                continue
            project_cols.append(F.col(src).alias(dst))
        projected_parent = parent_df.select(*project_cols).dropDuplicates([merge_col])

        parent_for_join = projected_parent
        if should_broadcast(parent_for_join):
            parent_for_join = F.broadcast(parent_for_join)

        joined = child_df.join(
            parent_for_join,
            on=merge_col,
            how="left",
        )

        # Fill defaults for nullable columns.
        if default_exprs:
            for col_name, lit in default_exprs.items():
                joined = joined.withColumn(
                    col_name,
                    F.when(F.col(col_name).isNull(), lit).otherwise(F.col(col_name)),
                )

        progress_callback(len(features) / float(max(self.num_features, 1)))
        return joined

    # ------------------------------------------------------------------
    # AggregationFeature — child→parent groupBy/agg
    # ------------------------------------------------------------------
    def _calculate_agg_features(
        self, features, frame, df_trie, progress_callback
    ):
        from pyspark.sql import functions as F

        test_feature = features[0]
        # Lazy plan for the child dataframe is stored at the end of the
        # relationship path in df_trie.
        base_frame = df_trie.get_node(test_feature.relationship_path).value
        if base_frame is None:
            raise ValueError(
                f"No child frame found in trie for feature {test_feature.get_name()} "
                f"along relationship path {test_feature.relationship_path}"
            )

        # Drop features whose output column is already present (mirrors
        # feature_set_calculator.py:660).
        existing_cols = set(frame.columns)
        features = [
            f
            for f in features
            if not all(n in existing_cols for n in f.get_feature_names())
        ]
        if not features:
            return frame

        # where-filter
        where = test_feature.where
        if where is not None:
            base_frame = base_frame.filter(F.col(where.get_name()))

        # use_previous filter (time-window scoped to the cutoff)
        use_previous = test_feature.use_previous
        if use_previous is not None:
            child_dataframe_name = test_feature.base_features[0].dataframe_name
            child_schema = self.entityset.schemas.get(child_dataframe_name)
            if child_schema is not None and child_schema.time_index:
                if use_previous.has_no_observations():
                    time_first = self.time_last - use_previous
                    base_frame = base_frame.filter(
                        F.col(child_schema.time_index) >= F.lit(time_first)
                    )
                else:
                    # "Last N observations" semantics — needs a window
                    # function per group. Deferred to v2.
                    raise NotImplementedError(
                        "use_previous with observation count is not yet "
                        "supported on the Spark backend; use a time-based "
                        "Timedelta instead."
                    )

        groupby_col = _get_groupby_col(test_feature, self.entityset)

        # Partition features into Tier-1 aggregations (handled via
        # ``.agg(...)``) and callable-based ones (dispatched via pandas_udf
        # grouped-agg).
        agg_exprs = []
        feature_default_values: Dict[str, "pyspark.sql.Column"] = {}  # noqa: F821
        for f in features:
            if len(f.base_features) != 1 or f.number_output_features != 1:
                # Multi-column or multi-output agg — route via GROUPED_AGG
                # pandas_udf. Only supported for spark_compatible=False path
                # in the dispatcher. Surface a clear error for now.
                raise NotImplementedError(
                    f"Multi-column/multi-output aggregation primitive "
                    f"{type(f.primitive).__name__} is not yet supported on "
                    "Spark. Use the pandas backend for this feature."
                )
            col_name = f.base_features[0].get_name()
            agg_exprs.append(
                aggregation_feature_expr(f, col_name).alias(f.get_name())
            )
            if f.default_value is not None:
                feature_default_values[f.get_name()] = default_to_spark_literal(
                    f.default_value
                )

        aggregated = base_frame.groupBy(F.col(groupby_col)).agg(*agg_exprs)

        # Join aggregated back to the parent frame.
        parent_schema = self.entityset.schemas[test_feature.parent_dataframe_name]
        parent_key = parent_schema.index
        # Rename the groupby column to match the parent's index for the join.
        aggregated = aggregated.withColumnRenamed(groupby_col, f"_ft_join_{parent_key}")

        join_expr = F.col(parent_key) == F.col(f"_ft_join_{parent_key}")
        frame = frame.join(aggregated, join_expr, "left").drop(
            f"_ft_join_{parent_key}"
        )

        # Apply default values for missing groups.
        for col_name, lit in feature_default_values.items():
            frame = frame.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), lit).otherwise(F.col(col_name)),
            )

        progress_callback(len(features) / float(max(self.num_features, 1)))
        return frame

    # ------------------------------------------------------------------
    # Ancestor linkage helper (used by deep aggregations)
    # ------------------------------------------------------------------
    def _add_ancestor_relationship_columns(
        self, child_df, parent_df, ancestor_rel_columns, relationship
    ):
        from pyspark.sql import functions as F

        rel_name = f"{relationship._parent_dataframe_name}"
        new_cols = [f"{rel_name}.{c}" for c in ancestor_rel_columns]

        col_map: Dict[str, str] = {
            relationship._parent_column_name: relationship._child_column_name
        }
        for child_col, parent_col in zip(new_cols, ancestor_rel_columns):
            col_map[parent_col] = child_col

        # Build the projection that will be joined into the child.
        project_cols = [
            F.col(src).alias(dst)
            for src, dst in col_map.items()
            if src in parent_df.columns
        ]
        merge_df = parent_df.select(*project_cols).dropDuplicates(
            [relationship._child_column_name]
        )
        joined = child_df.join(
            merge_df,
            on=relationship._child_column_name,
            how="left",
        )
        return joined, new_cols

    # ------------------------------------------------------------------
    # Column selection for query_by_values
    # ------------------------------------------------------------------
    def _necessary_columns(
        self, dataframe_name: str, feature_names: Set[str]
    ) -> List[str]:
        """Return the list of columns that must be kept after filtering.

        Mirrors ``feature_set_calculator.py:822``. Always includes:
        - the primary index
        - the time index
        - all foreign-key columns (because we don't know which forward
          relationships downstream features will need)
        - any identity feature's source column
        """
        schema = self.entityset.schemas[dataframe_name]
        keep: Set[str] = set()
        keep.add(schema.index)
        if schema.time_index:
            keep.add(schema.time_index)
        keep.update(schema.foreign_keys)

        for name in feature_names:
            f = self.feature_set.features_by_name[name]
            if isinstance(f, IdentityFeature):
                keep.add(f.column_name)

        # Ensure all columns exist in the dataframe.
        df_cols = set(self.entityset.dataframe_dict[dataframe_name].columns)
        return [c for c in keep if c in df_cols]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _get_groupby_col(agg_feature, entityset) -> str:
    """Return the column in the child dataframe that links back to the
    parent — i.e. the child's foreign-key for the last hop of the
    relationship path.
    """
    path = agg_feature.relationship_path
    # path is a RelationshipPath; last edge's child side is what we group on
    if len(path) == 0:
        raise ValueError("AggregationFeature with empty relationship path")
    last_edge = path[-1]
    _is_forward, relationship = last_edge
    return relationship._child_column_name

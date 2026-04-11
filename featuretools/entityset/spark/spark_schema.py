"""
Lightweight logical schema for Spark EntitySets.

``SparkLogicalSchema`` replaces woodwork (``.ww``) on the Spark path. Woodwork
on Spark was removed with PR #2705 and we are explicitly not reintroducing
it — it was the root cause of the leaky pandas-API-on-Spark behavior we want
to avoid. This class carries the minimum information DFS needs at planning
time: which column is the primary key, which column is the time index,
which columns are foreign keys, and a simple mapping from column name to a
string logical type tag (``"Integer"``, ``"Double"``, ``"Datetime"``,
``"Boolean"``, ``"Categorical"``, ``"NaturalLanguage"``, etc.).

The logical-type strings are deliberately backend-agnostic; the conversion
to Spark ``DataType`` happens in
:mod:`featuretools.computational_backends.spark.spark_utils`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


# Logical type tags understood by the Spark backend. These mirror the subset of
# woodwork logical types that actually affect DFS planning. Anything not listed
# here falls through to "Unknown" which is treated as pass-through by the
# primitive dispatcher.
KNOWN_LOGICAL_TYPES = frozenset(
    {
        "Integer",
        "IntegerNullable",
        "Double",
        "Boolean",
        "BooleanNullable",
        "Datetime",
        "Timedelta",
        "Categorical",
        "Ordinal",
        "NaturalLanguage",
        "EmailAddress",
        "URL",
        "PhoneNumber",
        "PostalCode",
        "CountryCode",
        "IPAddress",
        "LatLong",
        "Unknown",
    }
)


@dataclass
class SparkLogicalSchema:
    """Schema metadata for a single dataframe inside a SparkEntitySet.

    Attributes
    ----------
    name : str
        The dataframe name as registered on the EntitySet.
    index : str
        The primary key column. Must be unique within the dataframe.
    time_index : Optional[str]
        Column to treat as the primary time index for cutoff-time filtering
        and training-window semantics. ``None`` if the dataframe is
        time-independent.
    foreign_keys : Set[str]
        Set of column names that link to parent dataframes. Populated as
        :meth:`SparkEntitySet.add_relationship` is called.
    logical_types : Dict[str, str]
        Column name → logical-type tag from :data:`KNOWN_LOGICAL_TYPES`.
    last_time_index_col : Optional[str]
        Column holding the last-time-index values for this dataframe, if the
        user has pre-computed it. We do not auto-compute last-time-index on
        the Spark path in the first cut.
    secondary_time_indexes : Dict[str, list]
        Mapping of secondary-time-index column → list of columns to mask
        when filtering by that index. Matches the woodwork semantic on the
        pandas path.
    """

    name: str
    index: str
    time_index: Optional[str] = None
    foreign_keys: Set[str] = field(default_factory=set)
    logical_types: Dict[str, str] = field(default_factory=dict)
    last_time_index_col: Optional[str] = None
    secondary_time_indexes: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        unknown = {
            lt for lt in self.logical_types.values() if lt not in KNOWN_LOGICAL_TYPES
        }
        if unknown:
            # Don't error — downgrade silently to "Unknown" so user code keeps
            # working. The dispatcher will fall back to pandas_udf for any
            # primitive that cares about logical type.
            self.logical_types = {
                col: (lt if lt in KNOWN_LOGICAL_TYPES else "Unknown")
                for col, lt in self.logical_types.items()
            }

    def mark_foreign_key(self, column_name: str) -> None:
        self.foreign_keys.add(column_name)

    def is_time_indexed(self) -> bool:
        return self.time_index is not None

    def semantic_tags(self, column_name: str) -> Set[str]:
        """Return the subset of semantic tags recognized by DFS planning.

        Used by ``_necessary_columns`` in the Spark calculator to decide which
        columns must be kept after filtering. Mirrors
        ``feature_set_calculator.py:_necessary_columns`` which reads
        ``df.ww.semantic_tags`` on the pandas path.
        """
        tags: Set[str] = set()
        if column_name == self.index:
            tags.add("index")
        if column_name in self.foreign_keys:
            tags.add("foreign_key")
        if column_name == self.time_index:
            tags.add("time_index")
        return tags

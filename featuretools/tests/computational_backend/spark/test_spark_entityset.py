"""
Unit tests for :class:`SparkEntitySet`.

These exercise the bits of the EntitySet that DFS touches at planning
time: ``add_dataframe``, ``add_relationship``, ``query_by_values``,
``_handle_time``, schema propagation, and graph traversal inherited from
the pandas parent class.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from featuretools.entityset.spark import SparkEntitySet  # noqa: E402
from featuretools.entityset.spark.spark_schema import SparkLogicalSchema  # noqa: E402


def test_schema_dataclass_defaults():
    s = SparkLogicalSchema(name="t", index="id")
    assert s.index == "id"
    assert s.time_index is None
    assert s.foreign_keys == set()
    assert s.logical_types == {}


def test_schema_downgrades_unknown_logical_type():
    s = SparkLogicalSchema(
        name="t", index="id", logical_types={"col": "FizzBuzz"}
    )
    # Unknown types silently downgrade to "Unknown" to keep user code working.
    assert s.logical_types["col"] == "Unknown"


def test_schema_semantic_tags():
    s = SparkLogicalSchema(
        name="t",
        index="id",
        time_index="ts",
        foreign_keys={"fk"},
    )
    assert "index" in s.semantic_tags("id")
    assert "time_index" in s.semantic_tags("ts")
    assert "foreign_key" in s.semantic_tags("fk")
    assert s.semantic_tags("other") == set()


# ----------------------------------------------------------------------
# Fixtures using spark_session (defined in conftest.py)
# ----------------------------------------------------------------------
def test_add_dataframe_requires_explicit_index(spark_session):
    import pandas as pd

    pdf = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    sdf = spark_session.createDataFrame(pdf)
    es = SparkEntitySet(id="test")
    with pytest.raises(ValueError, match="requires an explicit `index`"):
        es.add_dataframe(dataframe_name="t", dataframe=sdf)


def test_add_dataframe_rejects_non_spark_df():
    import pandas as pd

    pdf = pd.DataFrame({"id": [1, 2, 3]})
    es = SparkEntitySet(id="test")
    with pytest.raises(TypeError, match="requires a pyspark.sql.DataFrame"):
        es.add_dataframe(dataframe_name="t", dataframe=pdf, index="id")


def test_add_dataframe_registers_schema(spark_session):
    import pandas as pd

    pdf = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    sdf = spark_session.createDataFrame(pdf)
    es = SparkEntitySet(id="test")
    es.add_dataframe(dataframe_name="t", dataframe=sdf, index="id")
    assert "t" in es.dataframe_dict
    assert es.schemas["t"].index == "id"
    assert es.schemas["t"].time_index is None
    # Inferred logical types should contain entries for both columns.
    assert set(es.schemas["t"].logical_types.keys()) == {"id", "val"}


def test_add_relationship_validates_columns(spark_session):
    import pandas as pd

    parent = spark_session.createDataFrame(
        pd.DataFrame({"pid": [1, 2, 3]})
    )
    child = spark_session.createDataFrame(
        pd.DataFrame({"cid": [10, 20], "pid": [1, 2]})
    )
    es = SparkEntitySet(id="test")
    es.add_dataframe(dataframe_name="parent", dataframe=parent, index="pid")
    es.add_dataframe(dataframe_name="child", dataframe=child, index="cid")
    es.add_relationship("parent", "pid", "child", "pid")

    assert len(es.relationships) == 1
    assert es.schemas["child"].foreign_keys == {"pid"}

    # Missing parent column
    with pytest.raises(ValueError, match="not in dataframe"):
        es.add_relationship("parent", "nope", "child", "pid")


def test_add_relationship_requires_parent_index_column(spark_session):
    import pandas as pd

    parent = spark_session.createDataFrame(pd.DataFrame({"pid": [1, 2]}))
    child = spark_session.createDataFrame(
        pd.DataFrame({"cid": [10, 20], "pid": [1, 2]})
    )
    es = SparkEntitySet(id="test")
    es.add_dataframe(dataframe_name="parent", dataframe=parent, index="pid")
    es.add_dataframe(dataframe_name="child", dataframe=child, index="cid")
    # parent_column_name must equal parent.index
    with pytest.raises(ValueError, match="is not the index of"):
        es.add_relationship("parent", "cid", "child", "pid")


def test_query_by_values_filters_by_instance_vals(mock_customer_spark_es):
    es = mock_customer_spark_es
    result = es.query_by_values(
        "customers", instance_vals=[1, 2], column_name="customer_id"
    )
    rows = result.select("customer_id").collect()
    got = {r["customer_id"] for r in rows}
    assert got == {1, 2}


def test_query_by_values_applies_time_filter(mock_customer_spark_es):
    import pandas as pd

    es = mock_customer_spark_es
    # Use a cutoff that's before some but not all transactions.
    cutoff = pd.Timestamp("2014-01-01 01:00:00")
    result = es.query_by_values(
        "transactions",
        instance_vals=None,
        time_last=cutoff,
    )
    # All rows after the filter should satisfy the cutoff.
    rows = result.select("transaction_time").collect()
    assert all(r["transaction_time"] <= cutoff for r in rows)


def test_forward_and_backward_relationships(mock_customer_spark_es):
    es = mock_customer_spark_es
    # Reuses EntitySet's graph traversal — should work unchanged.
    forward = es.get_forward_relationships("transactions")
    assert len(forward) >= 1
    backward = es.get_backward_relationships("customers")
    assert len(backward) >= 1

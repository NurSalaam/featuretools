"""
Pytest fixtures for the native-Spark DFS backend tests.

All fixtures here ``pytest.importorskip("pyspark")`` so the tests silently
skip on machines without a Spark install. The GitHub Actions ``test-spark``
job installs ``pip install -e '.[spark,test]'`` with Java 17 so these
fixtures materialize.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def spark_session():
    """Session-scoped local Spark session.

    We use ``local[2]`` (two executor threads) so shuffle/aggregation paths
    actually exercise more than one partition. ``arrow.enabled`` gets the
    pandas_udf fallback path onto Arrow.
    """
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("featuretools-spark-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def mock_customer_pandas_es():
    """The bundled ``mock_customer`` pandas EntitySet. Reused here so both
    backends are tested against identical inputs."""
    import featuretools as ft

    return ft.demo.load_mock_customer(return_entityset=True)


@pytest.fixture
def mock_customer_spark_es(spark_session, mock_customer_pandas_es):
    """A SparkEntitySet mirror of ``mock_customer_pandas_es``.

    Each pandas DataFrame is converted to a Spark DataFrame (with explicit
    schema inference) and registered with the same index/time_index as the
    pandas version. Relationships are added in the same order.
    """
    from featuretools.entityset.spark import SparkEntitySet

    pandas_es = mock_customer_pandas_es
    dataframes = {}
    for name, df in pandas_es.dataframe_dict.items():
        pdf = df.copy()
        # Drop woodwork accessor state since Spark doesn't use it.
        pdf = pd.DataFrame(pdf)
        spark_df = spark_session.createDataFrame(pdf)
        dataframes[name] = (
            spark_df,
            df.ww.index,
            df.ww.time_index,
        )

    relationships = [
        (
            r._parent_dataframe_name,
            r._parent_column_name,
            r._child_dataframe_name,
            r._child_column_name,
        )
        for r in pandas_es.relationships
    ]
    return SparkEntitySet(
        id="mock_customer_spark",
        dataframes=dataframes,
        relationships=relationships,
    )

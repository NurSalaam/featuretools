# flake8: noqa
from featuretools.entityset.deserialize import read_entityset
from featuretools.entityset.entityset import EntitySet
from featuretools.entityset.relationship import Relationship
from featuretools.entityset.timedelta import Timedelta

# Guarded re-export of the native-Spark EntitySet. Only importable when
# ``pyspark`` is installed (the Spark dependency is opt-in via
# ``pip install featuretools[spark]``).
try:
    from featuretools.entityset.spark.spark_entityset import SparkEntitySet
except ImportError:
    SparkEntitySet = None  # type: ignore[assignment]

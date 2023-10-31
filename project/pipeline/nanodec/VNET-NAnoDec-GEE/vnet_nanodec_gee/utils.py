import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import psutil
import pyspark.sql.dataframe
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime


def patch_time_windows(df: pyspark.sql.dataframe, window_seconds: int):
    """
    Generate time window by
    :param df: pyspark dataframe
    :param window_seconds: window size in second
    :type df: pyspark.sql.dataframe
    :type window_seconds: int
    :return: df
    :rtype: pyspark.sql.dataframe
    """
    time_window = from_unixtime(col('timestamp') - col('timestamp') % window_seconds)

    df = (
        df
            .withColumn('time_window', time_window)
    )

    return df


def init_local_spark(additional_spark_config: Tuple[Tuple[str, str]]):
    # initialise local spark
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    memory_gb = psutil.virtual_memory().available // (1024 ** 3)

    spark = (
        SparkSession
            .builder
            .master('local[*]')
            .config('spark.driver.memory', f'{memory_gb}g')
    )

    for name, value in additional_spark_config:
        spark = spark.config(name, value)

    spark = (
        spark
            .config('spark.driver.host', '127.0.0.1')
            .getOrCreate()
    )

    return spark


def normalise(x: float, min_val: float, max_val: float) -> float:
    norm_x = (x - min_val) / (max_val - min_val)
    if norm_x < 0:
        norm_x = 0.0
    elif norm_x > 1.0:
        norm_x = 1.0

    return norm_x


def row_generator(x):
    time_window, src_ip, feature, label = x
    return {
        'time_window': time_window,
        'src_ip': src_ip,
        'feature': np.expand_dims(np.array(feature, dtype=np.float32), axis=0),
        'label': label,
    }


def change_df_schema(spark: SparkSession, schema: Unischema, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    rows_rdd = (
        df
            .rdd
            .map(row_generator)
            .map(lambda x: dict_to_spark_row(schema, x))
    )

    df = spark.createDataFrame(
        rows_rdd,
        schema.as_spark_schema()
    )

    return df


def save_parquet_for_petastorm_parquet(spark: SparkSession, df: pyspark.sql.DataFrame, output_path: str,
                                       schema: Unischema):
    output_path = Path(output_path).absolute().as_uri()
    with materialize_dataset(spark, output_path, schema, row_group_size_mb=256):
        (
            df
                .write
                .mode('overwrite')
                .parquet(output_path)
        )


def get_uri(path):
    uri = Path(path).absolute().as_uri()

    if os.name == 'nt':
        return re.sub(r'^file:///', r'file://', uri)
    else:
        return uri

import logging
from typing import Optional, Tuple
import warnings
from warnings import simplefilter

import click
import numpy as np
import pandas as pd
from petastorm.codecs import CompressedNdarrayCodec, ScalarCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql.functions import col, when
import pyspark.sql.types
from pyspark.sql.types import StructType, StringType
from pyspark.ml.feature import VectorAssembler
from petastorm.etl.dataset_metadata import materialize_dataset

from .utils import get_uri, init_local_spark
from . import config


simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


VECTOR_ASSEMBLER_COL_NAME = 'vfeatures'


def process_dataset(
        data_dirpath, target_data_dirpath, spark, schema, unischema, columns_to_scale, vector_assembler, logger,
        scale=True, min_max_coefs=None, clip=True, use_labels=True, filter_non_background=True, row_group_size_mb=256,
):
    df = (
        spark
        .read
        .option('header', True)
        .schema(schema)
        .csv(get_uri(data_dirpath))
    )

    df = handle_missing_values(df, use_labels=use_labels)

    if scale and min_max_coefs is None:
        min_max_coefs = get_scaling_coefficients(df)

    if scale:
        df = scale_data(df, columns_to_scale, min_max_coefs)
    if clip:
        df = clip_data_to_0_1(df, columns_to_scale)
    if use_labels and filter_non_background:
        df = filter_non_background_data(df)

    # create feature vector (a single array of feature values)
    df = vector_assembler.transform(df)

    df = df.select(VECTOR_ASSEMBLER_COL_NAME, 'Label', 'IP_SRC', 'IP_DST')

    # redefine dataframe schema for petastorm dataframe
    rows_rdd = (
        df
        .rdd
        .map(row_generator)
        .map(lambda x: dict_to_spark_row(unischema, x))
    )

    df = spark.createDataFrame(
        rows_rdd,
        unischema.as_spark_schema()
    )

    logger.info('Persisting as petastorm files...')
    with materialize_dataset(spark, get_uri(target_data_dirpath), unischema, row_group_size_mb=row_group_size_mb):
        (
            df
            .write
            .mode('overwrite')
            .parquet(get_uri(target_data_dirpath))
        )

    return min_max_coefs


def row_generator(x):
    feature, label, src_ip, dst_ip = x
    return {
        'feature': np.expand_dims(np.array(feature, dtype=np.float32), axis=0),
        'label': label,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
    }


def get_scaling_coefficients(df_train):
    min_max_coefs = df_train.summary('min', 'max').drop(
        *[column.name for column in config.VNET_COLUMNS if not column.column_type.startswith('feature_')]
    ).toPandas().to_dict(orient='list')

    if 'summary' in min_max_coefs:
        del min_max_coefs['summary']

    for k, v in min_max_coefs.items():
        if None not in v:
            min_max_coefs[k] = [float(i) for i in v]
        else:
            warnings.warn(f'Feature "{k}" contains invalid values')
            min_max_coefs[k] = [0.0] * len(v)

    return min_max_coefs


def scale_data(df, columns, min_max_coefs):
    for column in columns:
        df = df.withColumn(
            column,
            ((col(column) - min_max_coefs.get(column)[0]) / (min_max_coefs.get(column)[1] - min_max_coefs.get(column)[0])))
    return df


def clip_data_to_0_1(df, columns):
    for column in columns:
        df = df.withColumn(column, when(df[column] < 0, 0.0).otherwise(df[column]))
        df = df.withColumn(column, when(df[column] > 1, 1.0).otherwise(df[column]))
    return df


def handle_missing_values(df, use_labels):
    if use_labels:
        df = df.filter(col('Label').isNotNull())
    df = df.na.fill(0.0)
    return df


def filter_non_background_data(df):
    df = df.filter(col('Label').contains('background'))
    return df


@click.command()
@click.option('--train', help='path to the directory containing train csv files', required=False)
@click.option('--validate', help='path to the directory containing validation csv files', required=False)
@click.option('--test', help='path to the directory containing test csv files', required=False)
@click.option('--target_train',
              help='path to the directory to persist train model input files (petastorm parquet format)', required=False)
@click.option('--target_validate',
              help='path to the directory to persist validation model input files (petastorm parquet format)', required=False)
@click.option('--target_test',
              help='path to the directory to persist test model input files (petastorm parquet format)', required=False)
@click.option('-s', '--spark_config',
              help='additional Spark configuration, specified as <name> <value> pairs', nargs=2, type=str, multiple=True)
@click.option('--scale', help='if specified, scale data to 0-1 range', is_flag=True)
@click.option('--clip', help='if specified, clip data to 0-1', is_flag=True)
@click.option('--add_labels_to_test', help='if specified, add labels to test data (applies to --test only)', is_flag=True)
def main(
        train: Optional[str],
        validate: Optional[str],
        test: Optional[str],
        target_train: Optional[str],
        target_validate: Optional[str],
        target_test: Optional[str],
        spark_config: Tuple[Tuple[str, str]],
        scale: bool,
        clip: bool,
        add_labels_to_test: bool,
):
    logger = logging.getLogger(__file__)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Initialising local spark')
    spark = init_local_spark(spark_config)

    schema = StructType()
    for column in config.VNET_COLUMNS:
        schema = schema.add(column.name, getattr(pyspark.sql.types, column.data_type)(), True)

    feature_names = [column.name for column in config.VNET_COLUMNS if column.column_type.startswith('feature_')]
    columns_to_scale = [column.name for column in config.VNET_COLUMNS if column.should_scale]

    unischema = Unischema(
        'data_schema', [
            UnischemaField('feature', np.float32, (1, len(feature_names)), CompressedNdarrayCodec(), False),
            UnischemaField('label', str, (), ScalarCodec(StringType()), True),
            UnischemaField('src_ip', str, (), ScalarCodec(StringType()), True),
            UnischemaField('dst_ip', str, (), ScalarCodec(StringType()), True),
        ]
    )

    vector_assembler = VectorAssembler(inputCols=feature_names, outputCol=VECTOR_ASSEMBLER_COL_NAME)
    min_max_scaling_coefs = None

    if train:
        logger.info('Processing train files')

        min_max_scaling_coefs = process_dataset(
            train, target_train, spark, schema, unischema, columns_to_scale, vector_assembler, logger,
            scale=scale, min_max_coefs=None, clip=clip, use_labels=True,
        )

        logger.info('Train input done')

    if validate:
        logger.info('Processing validation files')

        process_dataset(
            validate, target_validate, spark, schema, unischema, columns_to_scale, vector_assembler, logger,
            scale=scale, min_max_coefs=min_max_scaling_coefs, clip=clip, use_labels=True,
        )

        logger.info('Validation input done')

    if test:
        logger.info('Processing test files')

        process_dataset(
            test, target_test, spark, schema, unischema, columns_to_scale, vector_assembler, logger,
            scale=scale, min_max_coefs=min_max_scaling_coefs, clip=clip, use_labels=add_labels_to_test, filter_non_background=False,
        )

        logger.info('Test input done')


if __name__ == '__main__':
    main()

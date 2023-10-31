import logging
from typing import Tuple

import click
import pyspark
import numpy as np
from petastorm.codecs import CompressedNdarrayCodec, ScalarCodec
from petastorm.unischema import Unischema, UnischemaField
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

from utils import init_local_spark, normalise, change_df_schema, save_parquet_for_petastorm_parquet
import config


feature_min_max = {
    'mean_duration': (0.0, 2042.86),
    'mean_packet': (1.0, 109214.27272727272),
    'mean_num_of_bytes': (28.0, 163795638.0909091),
    'mean_packet_rate': (0.0, 17224.14377310265),
    'mean_byte_rate': (0.0, 13902452.340182647),
    'std_duration': (0.0, 562.7625560888366),
    'std_packet': (0.0, 370614.95468242496),
    'std_num_of_bytes': (0.0, 543247494.7844237),
    'std_packet_rate': (0.0, 15783.66319664221),
    'std_byte_rate': (0.0, 16441139.793386225),
    'entropy_protocol': (0.0, 2.260220915066596),
    'entropy_dst_ip': (0.0, 13.787687869067254),
    'entropy_src_port': (0.0, 14.206227931544092),
    'entropy_dst_port': (0.0, 14.027301292191831),
    'entropy_flags': (0.0, 4.631615665225586)
}


class FeatureComposer:
    def __init__(self, spark: SparkSession, df: pyspark.sql.DataFrame):
        self.spark = spark
        self.df = df
        self.feature_column = (
            'mean_duration', 'mean_packet', 'mean_num_of_bytes', 'mean_packet_rate', 'mean_byte_rate', 'std_duration',
            'std_packet', 'std_num_of_bytes', 'std_packet_rate', 'std_byte_rate', 'entropy_protocol', 'entropy_dst_ip',
            'entropy_src_port', 'entropy_dst_port', 'entropy_flags', 'proportion_src_port', 'proportion_dst_port',
        )

        self.feature_compose_udf = udf(self.feature_compose, 'array<double>')

    @staticmethod
    def feature_compose(
            mean_duration: float, mean_packet: float, mean_num_of_bytes: float, mean_packet_rate: float,
            mean_byte_rate: float, std_duration: float, std_packet: float, std_num_of_bytes: float,
            std_packet_rate: float, std_byte_rate: float, entropy_protocol: float, entropy_dst_ip: float,
            entropy_src_port: float, entropy_dst_port: float, entropy_flags: float, proportion_src_port: list,
            proportion_dst_port: list
    ) -> list:
        """
        Compose the feature array
        :param mean_duration: mean duration
        :param mean_packet: mean packet
        :param mean_num_of_bytes: mean number of bytes
        :param mean_packet_rate: mean packet rate
        :param mean_byte_rate: mean byte rate
        :param std_duration: std duration
        :param std_packet: std packet
        :param std_num_of_bytes: std number of bytes
        :param std_packet_rate: std packet rate
        :param std_byte_rate: std byte rate
        :param entropy_protocol: entropy of protocol
        :param entropy_dst_ip: entropy of dest ip
        :param entropy_src_port: entropy of src ip
        :param entropy_dst_port: entropy of dest port
        :param entropy_flags: entropy of flags
        :param proportion_src_port: proportion of src common ports
        :param proportion_dst_port: proportion of dest common port
        :type mean_duration: float
        :type mean_packet: float
        :type mean_num_of_bytes: float
        :type mean_packet_rate: float
        :type mean_byte_rate: float
        :type std_duration: float
        :type std_packet: float
        :type std_num_of_bytes: float
        :type std_packet_rate: float
        :type std_byte_rate: float
        :type entropy_protocol: float
        :type entropy_dst_ip: float
        :type entropy_src_port: float
        :type entropy_dst_port: float
        :type entropy_flags: float
        :type proportion_src_port: list
        :type proportion_dst_port: list
        :return: feature array
        :rtype list
        """
        # normalise
        mean_duration = normalise(mean_duration, *feature_min_max.get('mean_duration'))
        mean_packet = normalise(mean_packet, *feature_min_max.get('mean_packet'))
        mean_num_of_bytes = normalise(mean_num_of_bytes, *feature_min_max.get('mean_num_of_bytes'))
        mean_packet_rate = normalise(mean_packet_rate, *feature_min_max.get('mean_packet_rate'))
        mean_byte_rate = normalise(mean_byte_rate, *feature_min_max.get('mean_byte_rate'))
        std_duration = normalise(std_duration, *feature_min_max.get('std_duration'))
        std_packet = normalise(std_packet, *feature_min_max.get('std_packet'))
        std_num_of_bytes = normalise(std_num_of_bytes, *feature_min_max.get('std_num_of_bytes'))
        std_packet_rate = normalise(std_packet_rate, *feature_min_max.get('std_packet_rate'))
        std_byte_rate = normalise(std_byte_rate, *feature_min_max.get('std_byte_rate'))
        entropy_protocol = normalise(entropy_protocol, *feature_min_max.get('entropy_protocol'))
        entropy_dst_ip = normalise(entropy_dst_ip, *feature_min_max.get('entropy_dst_ip'))
        entropy_src_port = normalise(entropy_src_port, *feature_min_max.get('entropy_src_port'))
        entropy_dst_port = normalise(entropy_dst_port, *feature_min_max.get('entropy_dst_port'))
        entropy_flags = normalise(entropy_flags, *feature_min_max.get('entropy_flags'))

        feature_arr = [
            mean_duration, mean_packet, mean_num_of_bytes, mean_packet_rate, mean_byte_rate, std_duration, std_packet,
            std_num_of_bytes, std_packet_rate, std_byte_rate, entropy_protocol, entropy_dst_ip, entropy_src_port,
            entropy_dst_port, entropy_flags,
        ]

        feature_arr.extend(proportion_src_port)
        feature_arr.extend(proportion_dst_port)

        return feature_arr

    def transform(self, remove_malicious=True, remove_null_label=True) -> pyspark.sql.DataFrame:
        df = (
            self.df
                # compose feature
                .withColumn('features', self.feature_compose_udf(*self.feature_column))
        )

        if remove_null_label:
            df = df.filter(col('label').isNotNull())

        if remove_malicious:
            df = df.filter(col('label') == 'background')

        # select only time_window, src_ip, feature and label columns
        df = df.select(
            'time_window', 'src_ip', 'features', 'label',
        )

        return df


@click.command()
@click.option('--train', help='path to the directory containing train feature parquet files', required=True)
@click.option('--test', help='path to the directory containing test feature parquet files', required=True)
@click.option('--target_train', help='path to the directory to persist train model input files', required=True)
@click.option('--target_test', help='path to the directory to persist test model input files', required=True)
@click.option('-s', '--spark_config', help='additional Spark configuration, specified as <name> <value> pairs', nargs=2, type=str, multiple=True)
def main(train: str, test: str, target_train: str, target_test: str, spark_config: Tuple[Tuple[str, str]]):
    # initialise logger
    logger = logging.getLogger(__file__)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Initialising local spark')
    spark = init_local_spark(spark_config)

    logger.info('Preparing schema')
    # petastorm schema
    schema = Unischema(
        'data_schema', [
            UnischemaField('time_window', np.str, (), ScalarCodec(StringType()), False),
            UnischemaField('src_ip', np.str, (), ScalarCodec(StringType()), False),
            UnischemaField('feature', np.float32, (1, config.NUM_FEATURES_GEE), CompressedNdarrayCodec(), False),
            UnischemaField('label', np.str, (), ScalarCodec(StringType()), True),
        ]
    )

    # processing train
    logger.info('Processing train parquet files')
    logger.info('Read parquet')
    train_feature_df = spark.read.parquet(train)

    logger.info('Composing features...')
    train_input = FeatureComposer(spark, train_feature_df).transform(remove_malicious=True, remove_null_label=True)

    logger.info('Changing schema...')
    train_input = change_df_schema(spark, schema, train_input)

    logger.info('Persisting...')
    save_parquet_for_petastorm_parquet(spark, train_input, target_train, schema)

    logger.info('Train input done')

    # processing test
    logger.info('Processing test parquet files')
    logger.info('Read parquet')
    test_feature_df = spark.read.parquet(test)

    logger.info('Composing features...')
    test_input = FeatureComposer(spark, test_feature_df).transform(remove_malicious=False, remove_null_label=True)

    logger.info('Changing schema...')
    test_input = change_df_schema(spark, schema, test_input)

    logger.info('Persisting...')
    save_parquet_for_petastorm_parquet(spark, test_input, target_test, schema)

    logger.info('Test input done')


if __name__ == '__main__':
    main()

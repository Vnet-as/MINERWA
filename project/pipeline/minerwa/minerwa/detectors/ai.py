import asyncio
import functools
from multiprocessing import Pool, Process, Queue
import os
import pickle

import filelock
import numpy as np
from oslo_config import cfg
import pandas as pd
from petastorm import make_reader
from petastorm.codecs import CompressedNdarrayCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.pytorch import DataLoader
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.ml.feature import VectorAssembler
import pyspark.pandas as ps
import pyspark.sql.types
from pyspark.sql.types import StructType, StringType
import torch
import yaml
import zmq

from vnet_nanodec.anomaly_detection.preprocessing import (
    data_preparer,
    scaler,
    vnet_feature_extraction,
    vnet_preprocessing
)
from vnet_nanodec.anomaly_detection.preprocessing.windowing.flow_processor_multiproc import process_flows
from vnet_nanodec.defines import FEATURES_CASTDICT
from vnet_nanodec_gee import config
from vnet_nanodec_gee.build_model_input_vnet import clip_data_to_0_1, row_generator
from vnet_nanodec_gee.evaluate_vae import calc_recon_loss
from vnet_nanodec_gee.ml.vae import VAE
from vnet_nanodec_gee.utils import get_uri, init_local_spark

from minerwa.model import FlowBase
from .base import DetectorBase


COLUMNS = (
    'IN_BYTES',
    'IN_PKTS',
    'PROTOCOL',
    'TCP_FLAGS',
    'L4_SRC_PORT',
    'IPV4_SRC_ADDR',
    'IPV6_SRC_ADDR',
    'L4_DST_PORT',
    'IPV4_DST_ADDR',
    'IPV6_DST_ADDR',
    'OUT_BYTES',
    'OUT_PKTS',
    'MIN_IP_PKT_LEN',
    'MAX_IP_PKT_LEN',
    'ICMP_TYPE',
    'MIN_TTL',
    'MAX_TTL',
    'DIRECTION',
    'FLOW_START_MILLISECONDS',
    'FLOW_END_MILLISECONDS',
    'SRC_FRAGMENTS',
    'DST_FRAGMENTS',
    'CLIENT_TCP_FLAGS',
    'SERVER_TCP_FLAGS',
    'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    'NUM_PKTS_OVER_1514_BYTES',
    'LONGEST_FLOW_PKT',
    'SHORTEST_FLOW_PKT',
    'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_PKTS',
    'OOORDER_IN_PKTS',
    'OOORDER_OUT_PKTS',
    'DURATION_IN',
    'DURATION_OUT',
    'TCP_WIN_MIN_IN',
    'TCP_WIN_MAX_IN',
    'TCP_WIN_MSS_IN',
    'TCP_WIN_SCALE_IN',
    'TCP_WIN_MIN_OUT',
    'TCP_WIN_MAX_OUT',
    'TCP_WIN_MSS_OUT',
    'TCP_WIN_SCALE_OUT',
    'FLOW_VERDICT',
    'SRC_TO_DST_IAT_MIN',
    'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG',
    'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN',
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STDDEV',
    'APPLICATION_ID',
    'Label',
    'SRC_DENY',
    'DST_DENY',
)

FEATURE_NAMES = [
    column.name for column in config.VNET_COLUMNS
    if column.column_type.startswith('feature_')
]

LOCK = filelock.FileLock('/var/lock/minerwa_ai.lock')


class AIDetector(DetectorBase):

    role: str = ''

    CFG_OPTS = (
        cfg.IntOpt('window_size', default=2),
        cfg.IntOpt('win_min_entries', default=2),
        cfg.IntOpt('win_min_cnt', default=5),
        cfg.IntOpt('win_max_cnt', default=200),
        cfg.IntOpt('win_timeout', default=700),
        cfg.IntOpt('flow_winspan_max_len', default=2000),
        cfg.IntOpt('samples_cnt', default=30),
        cfg.IntOpt('spark_memory', default=50),
        cfg.StrOpt('spark_temp', default='/tmp/spark'),
        cfg.StrOpt('filter_model_path', default='/etc/minerwa/filter_model.pkl')
    )

    @classmethod
    def setup_config(cls, conf, conf_group_name: str = None):
        conf.register_opts(cls.CFG_OPTS, conf_group_name)

    async def _choose_role(self):
        while True:
            try:
                with LOCK.acquire(blocking=False):
                    if self.role != 'processor':
                        if self.role == 'ingestor':
                            await self._unset_ingestor()
                        self.role = 'processor'
                        await self._setup_processor()
            except filelock._error.Timeout:
                if self.role != 'ingestor':
                    self.role = 'ingestor'
                    await self._setup_ingestor()
            await asyncio.sleep(30)

    async def _setup_processor(self):
        self._flow_cache = []
        q_in = Queue()
        q_out = Queue()
        p = Process(target=self._analyze_flows, args=(q_in, q_out))
        ctx = zmq.Context()
        self._zmq_sock = ctx.socket(zmq.PULL)
        self._zmq_sock.bind('ipc:///tmp/minerwa_ai.socket')
        p.start()
        while self.role == 'processor':
            self._flow_cache.append(self._zmq_sock.recv_pyobj())
            if len(self._flow_cache) == 50:
                q_in.put(self._flow_cache.copy())
                print(q_in.qsize())
                self._flow_cache = []
            try:
                print(q_out.get_nowait())
            except:
                pass
        p.join()

    async def _setup_ingestor(self):
        ctx = zmq.Context()
        self._zmq_sock = ctx.socket(zmq.PUSH)
        self._zmq_sock.connect('ipc:///tmp/minerwa_ai.socket')
        await super().run()

    async def _unset_ingestor(self):
        await self._broker.drain()

    async def run(self):
        asyncio.create_task(self._choose_role())

    def _cache_flow(self, flow: FlowBase):
        self._zmq_sock.send_pyobj((flow.id, {
            'IN_BYTES': flow.inBytes,
            'IN_PKTS': flow.inPkts,
            'PROTOCOL': flow.protocol,
            'TCP_FLAGS': flow.tcpFlags,
            'L4_SRC_PORT': flow.l4SrcPort,
            'IPV4_SRC_ADDR': flow.ipv4SrcAddr,
            'IPV6_SRC_ADDR': flow.ipv6SrcAddr,
            'L4_DST_PORT': flow.l4DstPort,
            'IPV4_DST_ADDR': flow.ipv4DstAddr,
            'IPV6_DST_ADDR': flow.ipv6DstAddr,
            'OUT_BYTES': flow.outBytes,
            'OUT_PKTS': flow.outPkts,
            'MIN_IP_PKT_LEN': flow.minIpPktLen,
            'MAX_IP_PKT_LEN': flow.maxIpPktLen,
            'ICMP_TYPE': flow.icmpType,
            'MIN_TTL': flow.minTtl,
            'MAX_TTL': flow.maxTtl,
            'DIRECTION': flow.direction,
            'FLOW_START_MILLISECONDS': flow.flowStartMilliseconds,
            'FLOW_END_MILLISECONDS': flow.flowEndMilliseconds,
            'SRC_FRAGMENTS': flow.srcFragments,
            'DST_FRAGMENTS': flow.dstFragments,
            'CLIENT_TCP_FLAGS': flow.clientTcpFlags,
            'SERVER_TCP_FLAGS': flow.serverTcpFlags,
            'SRC_TO_DST_AVG_THROUGHPUT': flow.srcToDstAvgThroughput,
            'DST_TO_SRC_AVG_THROUGHPUT': flow.dstToSrcAvgThroughput,
            'NUM_PKTS_UP_TO_128_BYTES': flow.numPktsUpTo128Bytes,
            'NUM_PKTS_128_TO_256_BYTES': flow.numPkts128To256Bytes,
            'NUM_PKTS_256_TO_512_BYTES': flow.numPkts256To512Bytes,
            'NUM_PKTS_512_TO_1024_BYTES': flow.numPkts512To1024Bytes,
            'NUM_PKTS_1024_TO_1514_BYTES': flow.numPkts1024To1514Bytes,
            'NUM_PKTS_OVER_1514_BYTES': flow.numPktsOver1514Bytes,
            'LONGEST_FLOW_PKT': flow.longestFlowPkt,
            'SHORTEST_FLOW_PKT': flow.shortestFlowPkt,
            'RETRANSMITTED_IN_PKTS': flow.retransmittedInPkts,
            'RETRANSMITTED_OUT_PKTS': flow.retransmittedOutPkts,
            'OOORDER_IN_PKTS': flow.ooorderInPkts,
            'OOORDER_OUT_PKTS': flow.ooorderOutPkts,
            'DURATION_IN': flow.durationIn,
            'DURATION_OUT': flow.durationOut,
            'TCP_WIN_MIN_IN': flow.tcpWinMinIn,
            'TCP_WIN_MAX_IN': flow.tcpWinMaxIn,
            'TCP_WIN_MSS_IN': flow.tcpWinMssIn,
            'TCP_WIN_SCALE_IN': flow.tcpWinScaleIn,
            'TCP_WIN_MIN_OUT': flow.tcpWinMinOut,
            'TCP_WIN_MAX_OUT': flow.tcpWinMaxOut,
            'TCP_WIN_MSS_OUT': flow.tcpWinMssOut,
            'TCP_WIN_SCALE_OUT': flow.tcpWinScaleOut,
            'FLOW_VERDICT': flow.flowVerdict,
            'SRC_TO_DST_IAT_MIN': flow.srcToDstIatMin,
            'SRC_TO_DST_IAT_MAX': flow.srcToDstIatMax,
            'SRC_TO_DST_IAT_AVG': flow.srcToDstIatAvg,
            'SRC_TO_DST_IAT_STDDEV': flow.srcToDstIatStddev,
            'DST_TO_SRC_IAT_MIN': flow.dstToSrcIatMin,
            'DST_TO_SRC_IAT_MAX': flow.dstToSrcIatMax,
            'DST_TO_SRC_IAT_AVG': flow.dstToSrcIatAvg,
            'DST_TO_SRC_IAT_STDDEV': flow.dstToSrcIatStddev,
            'APPLICATION_ID': flow.applicationId,
            'Label': '',
            'SRC_DENY': False,
            'DST_DENY': False,
        }))

    def _analyze_flows(self, queue, result_queue):
        self._windower_settings = {
            'win_min_entries': self.conf.win_min_entries,
            'win_min_cnt': self.conf.win_min_cnt,
            'win_timeout': self.conf.win_timeout,
            'flow_winspan_max_len': self.conf.flow_winspan_max_len,
            'samples_cnt': self.conf.samples_cnt,
            'win_max_cnt': self.conf.win_max_cnt
        }

        with open('/etc/minerwa/scaling_config.yaml', 'r') as f:
            scaling_config = yaml.safe_load(f)

        spark = init_local_spark((
            ('spark.driver.memory', f'{self.conf.spark_memory}g'),
            ('spark.executor.memory', f'{self.conf.spark_memory}g'),
            ('spark.local.dir', self.conf.spark_temp),
            ('spark.sql.optimizer.maxIterations', 500),
        ))
        spark.conf.set('spark.sql.execution.arrow.pyspark.enabled',
                       'true')

        vector_assembler = VectorAssembler(inputCols=FEATURE_NAMES,
                                           outputCol='vfeatures')

        schema_struct = StructType()
        for column in config.VNET_COLUMNS:
            schema = schema_struct.add(column.name,
                                       getattr(pyspark.sql.types, column.data_type)(),
                                       True)

        columns_to_scale = [
            column.name for column in config.VNET_COLUMNS
            if column.should_scale
        ]

        unischema = Unischema(
            'data_schema', [
                UnischemaField('feature', np.float32, (1, len(FEATURE_NAMES)), CompressedNdarrayCodec(), False),
                UnischemaField('label', str, (), ScalarCodec(StringType()), True),
                UnischemaField('src_ip', str, (), ScalarCodec(StringType()), True),
                UnischemaField('dst_ip', str, (), ScalarCodec(StringType()), True),
            ]
        )

        with open('/etc/minerwa/filter_model.pkl', 'rb') as f:
            filter_model = pickle.load(f)

        num_features = len([column for column in config.VNET_COLUMNS if column.column_type.startswith('feature_')])

        vae_model = VAE.load_from_checkpoint(checkpoint_path='/etc/minerwa/vae_model', map_location=torch.device('cpu'), n_features=num_features)
        vae_model.eval()

        while True:
            flows = queue.get()
            ids, flows = zip(*flows)
            data = {
                col: [v[col] for v in flows]
                for col in COLUMNS
            }

            df = pd.DataFrame(data)
            df = df.astype(FEATURES_CASTDICT)
            win_stats, _ = process_flows(df, vnet_feature_extraction.extract_features,
                                         self._windower_settings, self.conf.window_size, None, 10)
            df = df.join(win_stats)

            df = vnet_preprocessing.preprocess(df)
            df = scaler.scale_features(df, scaling_config['scaling_config'])
            df = spark.createDataFrame(df, schema=schema)
            df = df.na.fill(0.0)
            df = clip_data_to_0_1(df, columns_to_scale)
            df = vector_assembler.transform(df)
            df = df.select('vfeatures', 'Label', 'IP_SRC', 'IP_DST')

            rows_rdd = (
                df
                .rdd
                .map(row_generator)
                .map(functools.partial(dict_to_spark_row, unischema))
            )

            df = spark.createDataFrame(
                rows_rdd,
                unischema.as_spark_schema()
            )

            data_folder = '/tmp/parquet_data'
            with materialize_dataset(spark, get_uri(data_folder), unischema, row_group_size_mb=100):
                (
                    df
                    .write
                    .mode('overwrite')
                    .parquet(get_uri(data_folder))
                )

            reader = make_reader(
                get_uri(data_folder), reader_pool_type='process', workers_count=10, num_epochs=1
            )
            dataloader = DataLoader(reader, batch_size=300, shuffling_queue_capacity=0)

            for data in dataloader:
                x = data['feature']
                recon_x, mu, logvar = vae_model(x)
                mse_loss = calc_recon_loss(recon_x, num_features, x, loss_type='mse')
                # bce_loss = calc_recon_loss(recon_x, num_features, x, loss_type='bce')
                # bce_kd_loss = calc_recon_loss(recon_x, num_features, x, logvar, mu, loss_type='bce+kd')
                print(mse_loss, flush=True)

                data_x = np.squeeze(x.numpy())
                pred = filter_model.predict(data_x)
                prob = filter_model.predict_proba(data_x).max(axis=1)
                print(pred, prob, flush=True)
                print('---', flush=True)


    async def process_flow(self, flow: FlowBase):
        self._cache_flow(flow)



"""
Loading and parsing ISP-owned IP addresses and their manually annotated categories.
"""

import os
import sys
import ipaddress
from typing import Union

import pandas as pd

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


class IPCategories:

    def __init__(
            self,
            df_ip_categories: pd.DataFrame,
            max_prefixlen_diff_for_subnet_expansion: int = 5,
    ):
        self.ip_categories = df_ip_categories.drop_duplicates(ignore_index=True).copy()
        self.max_prefixlen_diff_for_subnet_expansion = max_prefixlen_diff_for_subnet_expansion

        self.ip_categories['host_or_subnet'] = 'host'
        self.ip_categories.loc[
            self.ip_categories['ip'].str.contains('/', regex=False), 'host_or_subnet'
        ] = 'subnet'

        self.ip_categories['ip_obj'] = None

        host_cond = self.ip_categories['host_or_subnet'] == 'host'
        subnet_cond = self.ip_categories['host_or_subnet'] == 'subnet'
        self.ip_categories.loc[host_cond, 'ip_obj'] = (
            self.ip_categories.loc[host_cond, 'ip'].apply(ipaddress.ip_address))
        self.ip_categories.loc[subnet_cond, 'ip_obj'] = (
            self.ip_categories.loc[subnet_cond, 'ip'].apply(ipaddress.ip_network))

        self.ip_categories['version'] = self.ip_categories['ip_obj'].apply(self._get_ip_version)

        self.ip_categories = self._expand_subnets_to_ips(self.ip_categories)

        # Cannot reuse host_cond or subnet_cond, new rows have been added by now
        ip_addresses = self.ip_categories.loc[
            self.ip_categories['host_or_subnet'] == 'host', 'ip_obj'].tolist()
        ip_networks = self.ip_categories.loc[
            self.ip_categories['host_or_subnet'] == 'subnet', 'ip_obj'].tolist()

        self.ipv4_set = set([
            ip_address for ip_address in ip_addresses
            if isinstance(ip_address, ipaddress.IPv4Address)])
        self.ipv6_set = set([
            ip_address for ip_address in ip_addresses
            if isinstance(ip_address, ipaddress.IPv6Address)])
        self.ip_set = self.ipv4_set | self.ipv6_set

        self.ipv4_networks = [
            ip_network for ip_network in ip_networks
            if isinstance(ip_network, ipaddress.IPv4Network)]
        self.ipv6_networks = [
            ip_network for ip_network in ip_networks
            if isinstance(ip_network, ipaddress.IPv6Network)]
        self.ip_networks = self.ipv4_networks + self.ipv6_networks

    @staticmethod
    def _get_ip_version(
            ip_obj: Union[
                ipaddress.IPv4Address, ipaddress.IPv6Address,
                ipaddress.IPv4Network, ipaddress.IPv6Network],
    ) -> int:
        if isinstance(ip_obj, ipaddress.IPv4Address) or isinstance(ip_obj, ipaddress.IPv4Network):
            return 4
        else:
            return 6

    def _expand_subnets_to_ips(self, df_ip_categories: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            df_ip_categories.apply(self._expand_subnets_to_ips_per_row, axis=1).tolist(),
            ignore_index=True)

    def _expand_subnets_to_ips_per_row(self, row: pd.Series) -> pd.Series:
        if row['version'] not in [4, 6]:
            raise ValueError('invalid IP address version, must be 4 or 6')

        should_expand = False

        if row['host_or_subnet'] == 'subnet':
            if row['version'] == 4:
                max_prefixlen = 32
            else:
                max_prefixlen = 128

            if max_prefixlen - row['ip_obj'].prefixlen <= self.max_prefixlen_diff_for_subnet_expansion:
                should_expand = True

        if should_expand:
            new_ips = [
                pd.Series([str(new_ip), row['type'], 'host', new_ip, row['version']], index=row.index)
                for new_ip in row['ip_obj']
            ]
            return pd.DataFrame(new_ips, columns=row.index.tolist())
        else:
            return row.to_frame().T


def load_ip_categories(ip_categories_filepaths: list[str]) -> IPCategories:
    if isinstance(ip_categories_filepaths, str):
        ip_categories_filepaths = [ip_categories_filepaths]

    df_ip_categories = pd.concat([
        utils.load_df(filepath, csv_sep=',') for filepath in ip_categories_filepaths
    ]).reset_index(drop=True)

    return IPCategories(df_ip_categories)

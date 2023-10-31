"""
Windowed model for on-the-go traffic statistics computation.
"""

from collections import defaultdict
import functools
import operator

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
from scipy.stats import entropy

import traffic_stats as tstats
import streaming_stats as sstats
from flow_features import FlowFeatures
from flowlist import FlowList
from sampling import ReservoirSampler


# Dictionary mapper to translate protocols into assigned IDs
L4_PROTO_ID_MAPPER = defaultdict(lambda: 1 << 4, {
    'tcp'   : 1 << 0,
    'udp'   : 1 << 1,
    'icmp'  : 1 << 2,
    'sctp'  : 1 << 3,
})


class OpenWindowData:
    """Class encapsulating all statistics that are collected within the window for each IP

    Need to be implemented as standard class instead of dataclass due to problems with mutable
    datatypes in dataclasses.
    For more info, see: https://www.python.org/dev/peps/pep-0557/#mutable-default-values"""

    def __init__(self) -> None:
        self.logs           : np.ndarray = np.zeros(1, tstats.STATS_NPDTYPE_WINDOW_OPEN)
        self.flowlist       : FlowList   = FlowList()
        self.sport_uniques  : set        = set()
        self.sport_samples  : list       = []
        self.dst_ip_uniques : set        = set()


class Windower:
    """Windowing mechanism for statistics computation from flow data."""

    def __init__(self, win_min_entries, win_min_cnt, win_timeout, flow_winspan_max_len, *,
        samples_cnt=30, win_max_cnt=86400) -> None:
        """Initializes the Windower instance, setups the local variables and settings and starts
        the first window

        Parameters:
            win_min_entries      -- The minimum number of entries within the window to consider it
            win_min_cnt          -- Minimum number of windows to compute statistics upon
            win_timeout          -- Number of windows for window timeouting
            flow_winspan_max_len -- Maximum length for flows spanning multiple windows to consider
                                    in inter-flow times computation
            samples_cnt          -- Number of samples for port entropy computation
            win_max_cnt          -- Maximum number of windows to summarize
        """

        # Internal structures
        self.win_hist    = {}
        self.win_curr    = {}
        self.win_id      = np.uint32(0)
        self.stats_cache = {}

        # Windower settings
        self.win_min_entries      = win_min_entries
        self.win_max_cnt          = win_max_cnt
        self.win_min_cnt          = win_min_cnt
        self.win_timeout          = win_timeout
        self.flow_winspan_max_len = flow_winspan_max_len
        self.samples_cnt          = samples_cnt


    def log(self, features: FlowFeatures) -> None:
        """Logs the flow features into the current window structures.

        Parameters:
            featrues -- Flow features to be logged into the current window"""
        src_ip          = features.ip_src                               # Flow source IP address
        flow_duration   = features.tstamp_end - features.tstamp_start   # Flod duration in usec

        # Create an entry in the current window if it does not exist yet
        if src_ip not in self.win_curr:
            self.win_curr[src_ip] = OpenWindowData()
            self.win_curr[src_ip].sport_samples = [0] * self.samples_cnt

        winstats        = self.win_curr[src_ip]             # Current window stats for the given IP
        flows_so_far    = winstats.logs[0]['flows_total']   # Number of flows so far
        flows_with_this = flows_so_far + 1                  # Number of flows with this included

        dur_avg_old = winstats.logs[0]['dur_avg']   # Flow duration avg before recomputation
        ppf_avg_old = winstats.logs[0]['ppf_avg']   # Packets-per-flow avg before recomputation
        bpf_avg_old = winstats.logs[0]['bpf_avg']   # Bytes-per-flow avg before recomputation
        bpp_avg_old = winstats.logs[0]['bpp_avg']   # Bytes-per-packet avg before recomputation
        bps_avg_old = winstats.logs[0]['bps_avg']   # Bytes-per-second avg before recomputation
        pps_avg_old = winstats.logs[0]['pps_avg']   # Packets-per-second avg before recomputation
        ttl_avg_old = winstats.logs[0]['ttl_avg']   # TTL average before recomputation
        hdr_pld_ratio_avg_old = winstats.logs[0]['hdr_payload_ratio_avg']

        dur_avg_new = sstats.streaming_mean(flow_duration, dur_avg_old, flows_with_this)
        ppf_avg_new = sstats.streaming_mean(features.packets, ppf_avg_old, flows_with_this)
        bpf_avg_new = sstats.streaming_mean(features.bytes, bpf_avg_old, flows_with_this)
        bpp_avg_new = sstats.streaming_mean(features.bpp, bpp_avg_old, flows_with_this)
        bps_avg_new = sstats.streaming_mean(features.bps, bps_avg_old, flows_with_this)
        pps_avg_new = sstats.streaming_mean(features.pps, pps_avg_old, flows_with_this)
        ttl_avg_new = sstats.streaming_mean(features.ttl, ttl_avg_old, flows_with_this)
        hdr_pld_ratio_avg_new = sstats.streaming_mean(sstats.hdr_payload_estimate(
                features.ip_src, features.proto, features.packets, features.bytes),
            hdr_pld_ratio_avg_old, flows_with_this)

        # Log the features that will be used directly
        winstats.logs[0]['flows_total']  += 1
        winstats.logs[0]['dur_total']    += flow_duration
        winstats.logs[0]['dur_avg']       = dur_avg_new
        winstats.logs[0]['pkt_total']    += features.packets
        winstats.logs[0]['ppf_avg']       = ppf_avg_new
        winstats.logs[0]['bytes_total']  += features.bytes
        winstats.logs[0]['bpf_avg']       = bpf_avg_new
        winstats.logs[0]['bpp_avg']       = bpp_avg_new
        winstats.logs[0]['bps_avg']       = bps_avg_new
        winstats.logs[0]['pps_avg']       = pps_avg_new
        winstats.logs[0]['proto_all_or'] |= L4_PROTO_ID_MAPPER[features.proto]
        winstats.logs[0]['hdr_payload_ratio_avg'] = hdr_pld_ratio_avg_new

        # Log into auxilliary data structures
        winstats.logs[0]['dur_std_aux'] = sstats.streaming_variance_aux(flow_duration,
            winstats.logs[0]['dur_std_aux'], dur_avg_old, dur_avg_new)
        winstats.logs[0]['ppf_std_aux'] = sstats.streaming_variance_aux(features.packets,
            winstats.logs[0]['ppf_std_aux'], ppf_avg_old, ppf_avg_new)
        winstats.logs[0]['bpf_std_aux'] = sstats.streaming_variance_aux(features.bytes,
            winstats.logs[0]['bpf_std_aux'], bpf_avg_old, bpp_avg_new)
        winstats.logs[0]['bpp_std_aux'] = sstats.streaming_variance_aux(features.bpp,
            winstats.logs[0]['bpp_std_aux'], bpp_avg_old, bpp_avg_new)
        winstats.logs[0]['bps_std_aux'] = sstats.streaming_variance_aux(features.bps,
            winstats.logs[0]['bps_std_aux'], bps_avg_old, bps_avg_new)
        winstats.logs[0]['pps_std_aux'] = sstats.streaming_variance_aux(features.pps,
            winstats.logs[0]['pps_std_aux'], pps_avg_old, pps_avg_new)

        winstats.logs[0]['ttl_avg'] = ttl_avg_new
        winstats.logs[0]['ttl_std_aux'] = sstats.streaming_variance_aux(features.ttl,
            winstats.logs[0]['ttl_std_aux'], ttl_avg_old, ttl_avg_new)

        winstats.logs[0]['flows_syn'] += 1 if features.flag_syn else 0
        winstats.logs[0]['flows_ack'] += 1 if features.flag_ack else 0
        winstats.logs[0]['flows_fin'] += 1 if features.flag_fin else 0

        # Compute other auxiliary statistics not directly in the numpy array
        # Compute number of opened flows
        winstats.flowlist.add(features.tstamp_start, features.tstamp_end)

        # Sample the source port number and log unique port
        ReservoirSampler.sample_stateless(features.port_src, winstats.sport_samples, self.samples_cnt,
            flows_so_far)
        winstats.sport_uniques.add(features.port_src)

        # Log into unique destination IP addresses
        winstats.dst_ip_uniques.add(features.ip_dst)


    def end_window(self, win_incr=1) -> None:
        """Ends a window and saves the statistics into respective history structures.

        Parameters:
            win_incr   -- Number of windows to increment - used for optimization when no flows are
                present for a longer time and thus many windows would be skipped."""

        # Iterate through every entry in the current window
        for ip_entry in self.win_curr:
            win_contents = self.win_curr[ip_entry]

            # Skip entries with too few logs
            if win_contents.logs[0]['flows_total'] >= self.win_min_entries:
                fin_stats    = np.zeros(1, tstats.STATS_NPDTYPE_WINDOW_COMPLETED)  # Finalized stats
                win_logs     = win_contents.logs           # Logged data in the processed window
                flow_cnt     = win_logs[0]['flows_total']  # Total number of flows
                sport_counts = 0                           # Source port counts for entropy comp
                itimes_avg, itimes_std, itimes_min, itimes_max = \
                    win_contents.flowlist.query_inter_times(self.flow_winspan_max_len)

                # Create an empty list for a given IP if this is the first history log
                if ip_entry not in self.win_hist:
                    self.win_hist[ip_entry] = []

                # Remove the first entry in the list if it expired
                # Does not effect functionality, just improves memory management
                if len(self.win_hist[ip_entry]) > 0 and abs(self.win_id -
                  self.win_hist[ip_entry][0]['window_id'][0]) > self.win_timeout:
                    self.win_hist[ip_entry].pop(0)

                # Finalize stats regarding general flow information
                fin_stats[0]['window_id']   = self.win_id
                fin_stats[0]['flows_total'] = win_logs[0]['flows_total']
                fin_stats[0]['flows_concurrent_max'] = win_contents.flowlist.query_concurrent()
                fin_stats[0]['flows_itimes_avg'] = itimes_avg
                fin_stats[0]['flows_itimes_std'] = itimes_std
                fin_stats[0]['flows_itimes_min'] = itimes_min
                fin_stats[0]['flows_itimes_max'] = itimes_max
                fin_stats[0]['dur_total'] = win_logs[0]['dur_total']
                fin_stats[0]['dur_avg'] = win_logs[0]['dur_avg']
                fin_stats[0]['dur_std'] = sstats.streaming_variance(win_logs[0]['dur_std_aux'], flow_cnt)

                # Finalize stats regarding packet and bytes counts
                fin_stats[0]['pkt_total'] = win_logs[0]['pkt_total']
                fin_stats[0]['ppf_avg'] = win_logs[0]['ppf_avg']
                fin_stats[0]['ppf_std'] = sstats.streaming_variance(win_logs[0]['ppf_std_aux'], flow_cnt)
                fin_stats[0]['bytes_total'] = win_logs[0]['bytes_total']
                fin_stats[0]['bpf_avg'] = win_logs[0]['bpf_avg']
                fin_stats[0]['bpf_std'] = sstats.streaming_variance(win_logs[0]['bpf_std_aux'], flow_cnt)
                fin_stats[0]['bpp_avg'] = win_logs[0]['bpp_avg']
                fin_stats[0]['bpp_std'] = sstats.streaming_variance(win_logs[0]['bpp_std_aux'], flow_cnt)
                fin_stats[0]['bps_avg'] = win_logs[0]['bps_avg']
                fin_stats[0]['bps_std'] = sstats.streaming_variance(win_logs[0]['bps_std_aux'], flow_cnt)
                fin_stats[0]['pps_avg'] = win_logs[0]['pps_avg']
                fin_stats[0]['pps_std'] = sstats.streaming_variance(win_logs[0]['pps_std_aux'], flow_cnt)

                # Finalize stats regarding ports, protocols and their flags
                sport_counts = pd.Series(win_contents.sport_samples).value_counts()

                fin_stats[0]['port_src_uniq_cnt'] = len(win_contents.sport_uniques)
                fin_stats[0]['port_src_entropy']  = entropy(sport_counts)
                fin_stats[0]['proto_all_or']      = win_logs[0]['proto_all_or']
                fin_stats[0]['flag_syn_ratio']    = win_logs[0]['flows_syn'] / flow_cnt
                fin_stats[0]['flag_ack_ratio']    = win_logs[0]['flows_ack'] / flow_cnt
                fin_stats[0]['flag_fin_ratio']    = win_logs[0]['flows_fin'] / flow_cnt

                # Finalize miscellaneous stats
                fin_stats[0]['ip_dst_uniq'] = len(win_contents.dst_ip_uniques)
                fin_stats[0]['hdr_payload_ratio_avg'] = win_logs[0]['hdr_payload_ratio_avg']
                fin_stats[0]['ttl_std'] = sstats.streaming_variance(win_logs[0]['ttl_std_aux'], flow_cnt)

                # Append the filled entry into the history list
                self.win_hist[ip_entry].append(fin_stats)

        # Create a new window
        self.win_curr = {}
        self.win_id  += win_incr

        # Flush the stats cache
        self.stats_cache = {}


    def retrieve_stats(self, ip: str) -> np.ndarray:
        """Retrieves statistics regarding the current window based on the logged data from the
        past.

        Parameters:
            ip IP address to retrieve statistics for"""

        valid_start_idx = 0             # Index of the first valid processed window
        all_win_stats   = None          # Shortcut for list of window statistics for given IP
        summary_stats   = np.zeros(1, tstats.STATS_NPDTYPE_WINDOW_COMPLETED)
        interwin_stats  = np.zeros(1, tstats.STATS_NPDTYPE_INTERWINDOW)

        # Return stats from cache if they were already computed this window
        if ip in self.stats_cache:
            return self.stats_cache[ip]

        if ip in self.win_hist:
            # Get rid of the outdated windows
            for idx in range(len(self.win_hist[ip])):
                if abs(self.win_id - self.win_hist[ip][idx]['window_id'][0]) > self.win_timeout:
                    valid_start_idx += 1
                else:
                    break

            self.win_hist[ip] = self.win_hist[ip][valid_start_idx:]
            all_win_stats     = self.win_hist[ip][-self.win_max_cnt:]

            # Check if the minimum number of window entries is satisfied
            if len(all_win_stats) > self.win_min_cnt:
                # Merge all window stats into 1 numpy array and compute statistics upon it
                stats_merged = np.concatenate(all_win_stats, axis=0)

                summary_stats  = self._compute_summary_stats(stats_merged)
                interwin_stats = self._compute_interwind_stats(stats_merged)

        # Merge the computed statistics
        result_stats = rfn.merge_arrays([summary_stats, interwin_stats], flatten=True)

        # Cache computed stats for current window optimizations
        self.stats_cache[ip] = result_stats

        return result_stats


    #####   PRIVATE INTERFACE   #####
    @staticmethod
    def _compute_summary_stats(win_stats: np.ndarray) -> np.ndarray:
        """Computes summary statistics of multiple windows by averaging and applying other
        operations.

        Parameters:
            win_stats Window stats to perform summarization upon"""

        # Convert to pandas for easier mean computation - otherwise, manual approach is needed
        win_stats_pd = pd.DataFrame(win_stats)

        # Compute mean of each column and manually fix ones that require others than mean
        win_stats_summary_pd = win_stats_pd.mean()

        win_stats_summary_pd['flows_concurrent_max'] = win_stats_pd['flows_concurrent_max'].max()
        win_stats_summary_pd['flows_itimes_min']     = win_stats_pd['flows_itimes_min'].min()
        win_stats_summary_pd['flows_itimes_max']     = win_stats_pd['flows_itimes_max'].max()
        win_stats_summary_pd['proto_all_or']         = functools.reduce(operator.__or__,
            win_stats_pd['proto_all_or'])

        # Convert back to numpy and return
        win_stats_summary = win_stats_summary_pd.to_numpy()

        return np.core.records.fromarrays(win_stats_summary,
            dtype=tstats.STATS_NPDTYPE_WINDOW_COMPLETED).reshape((1,))

    @staticmethod
    def _compute_interwind_stats(win_stats: np.ndarray) -> np.ndarray:
        """Computes interwindow statistics of multiple windows by applying standard deviation
        and other operations

        Parameters:
            win_stats Window stats to compute interwindows stats upon"""

        # In cotrast to summary statistics, these interwindow stats need to be computed manually
        interwind_stats = np.zeros(1, tstats.STATS_NPDTYPE_INTERWINDOW)
        windows_cnt  = win_stats.shape[0]           # No. of windows for interwindow stats
        windows_span = 0                            # Window span
        win_id_first = win_stats[0]['window_id']    # First processed window ID
        win_id_last  = win_stats[len(win_stats) - 1]['window_id']   # Last processed window ID

        if win_id_last > win_id_first:
            # Normal behavior - no counter overflow occurence
            windows_span = win_id_last - win_id_first + 1
        else:
            # Counter overflow has occured - normalize the span number (+1 due to overflow)
            windows_span = np.iinfo(np.uint32).max - win_id_first + win_id_last  + 2

        interwind_stats[0]['window_total']              = windows_cnt
        interwind_stats[0]['window_span']               = windows_span
        interwind_stats[0]['window_active_ratio']       = windows_cnt / windows_span
        interwind_stats[0]['flows_total_std']           = win_stats['flows_total'].std()
        interwind_stats[0]['flows_concurrent_max_std']  = win_stats['flows_concurrent_max'].std()
        interwind_stats[0]['flows_itimes_avg_std']      = win_stats['flows_itimes_avg'].std()
        interwind_stats[0]['flows_itimes_std_std']      = win_stats['flows_itimes_std'].std()
        interwind_stats[0]['flows_itimes_min_std']      = win_stats['flows_itimes_min'].std()
        interwind_stats[0]['flows_itimes_max_std']      = win_stats['flows_itimes_max'].std()
        interwind_stats[0]['dur_total_std']             = win_stats['dur_total'].std()
        interwind_stats[0]['dur_avg_std']               = win_stats['dur_avg'].std()
        interwind_stats[0]['dur_std_std']               = win_stats['dur_std'].std()
        interwind_stats[0]['pkt_total_std']             = win_stats['pkt_total'].std()
        interwind_stats[0]['ppf_avg_std']               = win_stats['ppf_avg'].std()
        interwind_stats[0]['ppf_std_std']               = win_stats['ppf_std'].std()
        interwind_stats[0]['bytes_total_std']           = win_stats['bytes_total'].std()
        interwind_stats[0]['bpf_avg_std']               = win_stats['bpf_avg'].std()
        interwind_stats[0]['bpf_std_std']               = win_stats['bpf_std'].std()
        interwind_stats[0]['bps_avg_std']               = win_stats['bps_avg'].std()
        interwind_stats[0]['bps_std_std']               = win_stats['bps_std'].std()
        interwind_stats[0]['bpp_avg_std']               = win_stats['bpp_avg'].std()
        interwind_stats[0]['bpp_std_std']               = win_stats['bpp_std'].std()
        interwind_stats[0]['port_src_uniq_cnt_std']     = win_stats['port_src_uniq_cnt'].std()
        interwind_stats[0]['port_src_entropy_std']      = win_stats['port_src_entropy'].std()
        interwind_stats[0]['flag_syn_ratio_std']        = win_stats['flag_syn_ratio'].std()
        interwind_stats[0]['flag_ack_ratio_std']        = win_stats['flag_ack_ratio'].std()
        interwind_stats[0]['flag_fin_ratio_std']        = win_stats['flag_fin_ratio'].std()
        interwind_stats[0]['ip_dst_uniq_std']           = win_stats['ip_dst_uniq'].std()
        interwind_stats[0]['hdr_payload_ratio_avg_std'] = win_stats['hdr_payload_ratio_avg'].std()
        interwind_stats[0]['ttl_std_std']               = win_stats['ttl_std'].std()

        return interwind_stats

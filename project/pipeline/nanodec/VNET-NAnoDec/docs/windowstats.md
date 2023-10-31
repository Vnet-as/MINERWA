# Flow Windowing Routine Statistics Description

### Background
- Statistics are aggregated based on source IP address
- They can be used solely on their own or as an addition to other data to enhance the accuracy of the classification process
- Two types of statistics are computed
  - **Window statistics** - for each window (features 1-30)
  - **Inter-window statistics** - between windows (31-60)
- Generally, there are various schemes of how to compute window statistics for the final output
  - Averages of all summarized windows
  - Weighted averages of all summarized windows, older having lesser weight
  - Current (last) window separately and other historical windows aggregated
- Minimum of 2 flows per window need to be present in order to consider a window valid
- Minimum of 2 windows need to be processed to be able to compute inter-window stats

### Windower Statistics

| No. | Identifier                  | Datatype     | Description                               |
| --- | --------------------------- | ------------ | ----------------------------------------- |
| 1   |  `flows_total`              | `np.uint32`  | Total number of flows                     |
| 2   | `flows_concurrent_max`      | `np.uint16`  | Number of maximum concurrent flows        |
| 3   | `flows_itimes_avg`          | `np.float32` | Average time between flows' starts        |
| 4   | `flows_itimes_std`          | `np.float32` | Standard deviation between flows' starts  |
| 5   | `flows_itimes_min`          | `np.float32` | Minimum time between two flow starts      |
| 6   | `flows_itimes_max`          | `np.float32` | Maximum time between two flow starts      |
| 7   | `dur_total`                 | `np.float32` | Sum of flow durations within the window   |
| 8   | `dur_avg`                   | `np.float32` | Flow duration average                     |
| 9   | `dur_std`                   | `np.float32` | Flow duration standard deviations         |
| 10  | `pkt_total`                 | `np.uint32`  | Total number of packets                   |
| 11  | `ppf_avg`                   | `np.float32` | Packets per flow average                  |
| 12  | `ppf_std`                   | `np.float32` | Packets per flow standard deviation       |
| 13  | `bytes_total`               | `np.uint64`  | Total number of bytes                     |
| 14  | `bpf_avg`                   | `np.float32` | Bytes per flow average                    |
| 15  | `bpf_std`                   | `np.float32` | Bytes per flow standard deviation         |
| 16  | `bpp_avg`                   | `np.float32` | Bytes per packet average                  |
| 17  | `bpp_std`                   | `np.float32` | Bytes per packet standard deviation       |
| 18  | `bps_avg`                   | `np.float32` | Bytes per second average                  |
| 19  | `bps_std`                   | `np.float32` | Bytes per second standard deviation       |
| 20  | `pps_avg`                   | `np.float32` | Packets per second average                |
| 21  | `pps_std`                   | `np.float32` | Packets per second standard deviation     |
| 22  | `port_src_uniq_cnt`         | `np.uint16`  | Number of unique source ports             |
| 23  | `port_src_entropy`          | `np.float32` | Source port entropy                       |
| 24  | `proto_all_or`              | `np.uint8`   | OR value of all encountered L4 protocols  |
| 25  | `flag_syn_ratio`            | `np.float32` | Ratio of SYN flags in flows               |
| 26  | `flag_ack_ratio`            | `np.float32` | Ratio of ACK flags in flows               |
| 27  | `flag_fin_ratio`            | `np.float32` | Ratio of FIN flags in flows               |
| 28  | `ip_dst_uniq`               | `np.uint16`  | Number of unique destination IP addresses |
| 29  | `hdr_payload_ratio_avg`     | `np.float32` | Packets' header to payload ratio estimate |
| 30  | `ttl_std`                   | `np.float32` | Time-to-live standard deviation           |
| 31  | `window_total`              | `np.uint16`  | Total number of summarized windows        |
| 32  | `window_span`               | `np.uint16`  | Last window ID - first window ID          |
| 33  | `window_active_ratio`       | `np.float32` | Windows ratio in which SRC IP was active* |
| 34  | `flows_total_std`           | `np.float32` | Stds of window stats within all winds ... |
| 34  | `flows_concurrent_max_std`  | `np.float32` | |
| 35  | `flows_itimes_avg_std`      | `np.float32` | |
| 37  | `flows_itimes_std_std`      | `np.float32` | |
| 38  | `flows_itimes_min_std`      | `np.float32` | |
| 39  | `flows_itimes_max_std`      | `np.float32` | |
| 40  | `dur_total_std`             | `np.float32` | |
| 41  | `dur_avg_std`               | `np.float32` | |
| 42  | `dur_std_std`               | `np.float32` | |
| 43  | `pkt_total_std`             | `np.float32` | |
| 44  | `ppf_avg_std`               | `np.float32` | |
| 45  | `ppf_std_std`               | `np.float32` | |
| 46  | `bytes_total_std`           | `np.float32` | |
| 47  | `bpf_avg_std`               | `np.float32` | |
| 48  | `bpf_std_std`               | `np.float32` | |
| 49  | `bps_avg_std`               | `np.float32` | |
| 50  | `bps_std_std`               | `np.float32` | |
| 51  | `bpp_avg_std`               | `np.float32` | |
| 52  | `bpp_std_std`               | `np.float32` | |
| 53  | `port_src_uniq_cnt_std`     | `np.float32` | |
| 54  | `port_src_entropy_std`      | `np.float32` | |
| 55  | `flag_syn_ratio_std`        | `np.float32` | |
| 56  | `flag_ack_ratio_std`        | `np.float32` | |
| 57  | `flag_fin_ratio_std`        | `np.float32` | |
| 58  | `ip_dst_uniq_std`           | `np.float32` | |
| 59  | `hdr_payload_ratio_avg_std` | `np.float32` | |
| 60  | `ttl_std_std`               | `np.float32` | |

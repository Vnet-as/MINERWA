#!/bin/bash

function runnprobe()
{	
	local fname="${1}"
	local job_prefix="${2}"
	local iter_start="$(date +%s)"
	local src_folder="${3}"
	local dst_folder="/data/kinit/flows/"

	#cca 5min
	unzstd -T4 -q /data/ai_train_pcaps_merged/$fname -o ${src_folder}${fname%.*}

	declare -r nprobe_ips="100.64.12.0/24,93.184.64.48/28,93.184.64.64/26,93.184.65.160/28,93.184.68.224/27,100.64.1.128/26,93.184.66.32/27,93.184.67.64/26,93.184.76.240/28,100.64.11.32/27,100.64.26.0/24,46.229.231.0/25,100.64.8.0/26,100.64.20.0/24,100.64.10.0/25,100.64.31.0/24,100.64.24.0/24,46.229.232.0/23,2a01:390:dc4::/64,217.73.25.32/27,86.110.251.0/26,86.110.251.64/26,100.64.34.0/24,81.89.63.192/26,109.74.152.0/25,100.64.27.0/26,100.64.27.128/25,100.64.11.64/26,81.89.58.0/25,81.89.62.0/24,217.73.22.0/24,81.89.58.192/28,100.64.4.64/26,100.64.8.128/26,100.64.22.0/24,185.184.28.0/24,100.64.8.64/26,93.184.67.0/26,86.110.249.0/24,86.110.245.216/29,100.64.23.0/24,100.64.4.0/27,100.64.4.32/27,100.64.11.128/26,217.73.21.0/27,100.64.11.0/27,100.64.15.0/24,81.89.52.128/25,81.89.53.0/26,81.89.57.128/25,93.184.67.128/25,100.64.4.128/25,100.64.28.0/25,100.64.5.0/25,100.64.10.128/25,46.229.227.80/28,100.64.33.0/24,86.110.232.192/28,100.64.7.0/24,100.64.29.0/24,100.64.25.0/24,46.229.227.160/28,86.110.232.32/27,86.110.245.32/28,100.64.32.0/24,100.64.8.192/26,92.240.228.0/24,92.240.229.0/24,92.240.230.0/24,92.240.231.0/24,92.240.234.0/24,92.240.235.0/24,92.240.241.0/24,92.240.242.0/24,92.240.244.0/24,92.240.245.0/24,92.240.249.0/24,92.240.253.0/24,92.240.254.0/24,92.240.255.0/24,2a00:10d8::/32,92.240.237.0/25,92.240.236.0/25,92.240.237.128/25,92.240.236.128/25,100.64.11.192/27,100.64.0.0/24,100.64.5.128/25,81.89.54.64/27,93.184.76.192/28,100.64.1.0/26,93.184.64.224/27,100.64.14.0/24,100.64.17.0/24,100.64.9.0/24,100.64.30.0/24,100.64.44.0/24,100.64.27.64/27,100.64.21.0/24,86.110.228.16/28,86.110.229.152/29,86.110.228.160/29,86.110.224.0/26,86.110.229.240/29,86.110.229.16/29,86.110.228.0/28,86.110.228.224/28,86.110.229.160/28,86.110.229.40/29,86.110.229.192/29,86.110.227.240/28,86.110.229.8/29,86.110.228.208/29,86.110.229.120/29,86.110.228.248/29,86.110.229.224/29,86.110.228.240/29,86.110.229.200/29,86.110.229.208/28,86.110.228.192/29,86.110.229.48/28,86.110.228.216/29,86.110.228.168/29,86.110.228.184/29,86.110.229.176/28,86.110.229.64/28,86.110.234.0/25,86.110.228.96/28,86.110.229.32/29,86.110.229.96/28,86.110.229.232/29,86.110.228.176/29,86.110.228.144/28,86.110.227.192/27,86.110.230.0/24,86.110.224.128/25,86.110.225.0/24,86.110.226.0/24,2a01:390:300::/64,86.110.228.128/28,86.110.228.200/29,86.110.228.32/28,86.110.228.48/28,86.110.229.128/28,86.110.228.112/28,86.110.234.128/26,86.110.227.32/27,86.110.228.64/27,86.110.232.0/24,86.110.231.0/24,86.110.227.224/28,86.110.227.96/27,86.110.227.0/27,86.110.227.128/26,86.110.227.64/27,100.64.6.0/24,100.64.18.0/24,46.229.227.112/28,46.229.234.0/24,81.89.48.0/24,81.89.49.0/24,86.110.240.192/27,86.110.247.96/28,93.184.68.128/26,93.184.69.0/24,93.184.70.0/24,93.184.79.0/24,109.74.149.0/24,109.74.154.0/24,217.73.17.0/24,217.73.31.0/24,86.110.248.0/24,93.184.75.0/24,93.184.71.0/24,93.184.77.0/24,109.74.153.0/24,109.74.156.0/24,109.74.157.0/24,193.28.8.0/24,109.74.146.0/24,2a01:390:1:e::/64,46.229.224.128/25,46.229.225.0/24,46.229.230.0/24,46.229.236.0/24,109.74.144.0/23,109.74.152.128/25,100.64.19.0/24,81.89.53.192/26,81.89.56.224/27,100.64.2.0/24,100.64.45.0/24,86.110.250.16/28,217.73.27.176/28,81.89.52.0/27,81.89.63.96/27,100.64.1.192/26,100.64.1.64/26,93.184.71.88/29,100.64.16.0/24,100.64.43.0/24,100.64.13.0/24,81.89.63.64/28,81.89.63.128/27,109.74.147.0/24,109.74.148.0/24,109.74.150.0/24,109.74.151.0/24,109.74.155.0/24,109.74.158.0/24,109.74.159.0/24,217.73.16.0/24,217.73.18.0/24,217.73.19.0/24,217.73.20.0/24,217.73.23.0/24,217.73.24.0/24,217.73.25.0/27,217.73.25.64/26,217.73.25.128/25,217.73.26.0/24,217.73.27.0/24,217.73.28.0/24,217.73.29.0/24,217.73.30.0/24,46.229.224.0/25,46.229.226.0/24,46.229.227.0/25,46.229.235.0/24,46.229.238.0/24,46.229.239.0/24,81.89.50.0/24,81.89.51.0/24,81.89.52.32/27,81.89.52.64/26,81.89.53.64/26,81.89.53.128/26,81.89.54.0/26,81.89.54.96/27,81.89.54.128/25,81.89.55.0/24,81.89.56.0/25,81.89.56.128/26,81.89.56.192/27,81.89.57.0/25,81.89.59.0/24,81.89.60.0/24,81.89.61.0/24,86.110.224.64/26,86.110.229.0/29,86.110.229.24/29,86.110.229.80/28,86.110.229.112/29,86.110.229.144/29,86.110.229.248/29,86.110.232.64/26,86.110.232.128/26,86.110.232.208/28,86.110.232.224/27,86.110.233.0/24,86.110.234.192/26,86.110.235.0/24,86.110.236.0/22,86.110.240.0/25,86.110.240.128/26,86.110.240.224/27,86.110.241.0/24,86.110.242.0/23,86.110.244.0/24,86.110.245.0/27,86.110.245.48/28,86.110.245.64/26,86.110.245.128/26,86.110.245.192/28,86.110.245.208/29,86.110.245.224/27,86.110.246.0/24,86.110.247.0/26,86.110.247.64/27,86.110.247.112/28,86.110.247.128/25,86.110.250.0/28,86.110.250.32/27,86.110.250.64/26,86.110.250.192/26,86.110.251.0/24,86.110.252.0/22,93.184.64.0/27,93.184.64.32/28,93.184.64.64/26,93.184.64.128/26,93.184.64.192/27,93.184.65.0/25,93.184.65.128/27,93.184.65.176/28,93.184.65.192/26,93.184.66.0/27,93.184.66.64/26,93.184.66.128/25,93.184.68.0/25,93.184.68.192/27,93.184.72.0/24,93.184.73.0/24,93.184.74.0/24,93.184.76.0/25,93.184.76.128/26,93.184.78.0/24,46.229.228.0/23,31.40.2.0/24,31.40.3.0/24,31.40.4.0/24,31.40.5.0/24,31.40.6.0/24,31.40.7.0/24,62.106.95.0/24,77.72.80.0/24,85.133.136.0/24,85.133.164.0/24,85.133.165.0/24,87.236.166.0/24,91.246.49.0/24,91.247.177.0/24,92.118.72.0/22,185.36.192.0/22,185.66.200.0/23,185.151.236.0/22,193.35.230.0/24,217.114.40.0/24,217.114.46.0/24"
	declare -r nprobe_template="%IN_BYTES %IN_PKTS %PROTOCOL %TCP_FLAGS %L4_SRC_PORT %IPV4_SRC_ADDR %IPV6_SRC_ADDR %L4_DST_PORT %IPV4_DST_ADDR %IPV6_DST_ADDR %OUT_BYTES %OUT_PKTS %MIN_IP_PKT_LEN %MAX_IP_PKT_LEN %ICMP_TYPE %MIN_TTL %MAX_TTL %DIRECTION %FLOW_START_MILLISECONDS %FLOW_END_MILLISECONDS %SRC_FRAGMENTS %DST_FRAGMENTS %CLIENT_TCP_FLAGS %SERVER_TCP_FLAGS %SRC_TO_DST_AVG_THROUGHPUT %DST_TO_SRC_AVG_THROUGHPUT %NUM_PKTS_UP_TO_128_BYTES %NUM_PKTS_128_TO_256_BYTES %NUM_PKTS_256_TO_512_BYTES %NUM_PKTS_512_TO_1024_BYTES %NUM_PKTS_1024_TO_1514_BYTES %NUM_PKTS_OVER_1514_BYTES %SRC_IP_COUNTRY %DST_IP_COUNTRY %SRC_IP_LONG %SRC_IP_LAT %DST_IP_LONG %DST_IP_LAT %LONGEST_FLOW_PKT %SHORTEST_FLOW_PKT %RETRANSMITTED_IN_PKTS %RETRANSMITTED_OUT_PKTS %OOORDER_IN_PKTS %OOORDER_OUT_PKTS %DURATION_IN %DURATION_OUT %TCP_WIN_MIN_IN %TCP_WIN_MAX_IN %TCP_WIN_MSS_IN %TCP_WIN_SCALE_IN %TCP_WIN_MIN_OUT %TCP_WIN_MAX_OUT %TCP_WIN_MSS_OUT %TCP_WIN_SCALE_OUT %FLOW_VERDICT %SRC_TO_DST_IAT_MIN %SRC_TO_DST_IAT_MAX %SRC_TO_DST_IAT_AVG %SRC_TO_DST_IAT_STDDEV %DST_TO_SRC_IAT_MIN %DST_TO_SRC_IAT_MAX %DST_TO_SRC_IAT_AVG %DST_TO_SRC_IAT_STDDEV %APPLICATION_ID"

	local pids=""; 

		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 1:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_1" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 2:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_2" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 4:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_3" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 16:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_4" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 32:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_5" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 64:1:1 --lifetime-timeout 60 --idle-timeout 30 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_6" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 1:1:1 --lifetime-timeout 30 --idle-timeout 15 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_7" & 
		pids="$pids $!"; 
		nprobe -n none -i none -V 10 -L ${nprobe_ips} -r -T "${nprobe_template}" --dump-format T --bi-directional -N 0 --dont-reforge-timestamps -u -1 -Q -1 -w 1200000 --max-num-flows 1000000 --max-log-lines 1000000 -S 1:1:1 --lifetime-timeout 2 --idle-timeout 2 -i "${src_folder}${fname%.*}" --dump-path "${dst_folder}scenario_8" & 
		pids="$pids $!"; 
		#remaining scenarios will use scenario_1 with different windowing during preproceessing
		
		echo -e "${job_prefix} nProbe running ..."
		
		#not enough - nprobe tends to hang after processing all packets from pcap, flows are still buffered
		#for pid in $pids; do wait $pid; done; 

		#wait over 30min and then end nprobe processes so they export even the buffered flows
		#sleep 2000
		#for pid in $pids; do kill -2 $pid; done;
		
		sleep 1200
		#maybe it would be better to check whether nprobe instance is still processing or hanging
		while :
		do
			sleep 30
			processing=false
			for pid in $pids;
			do
				if ps -p $pid > /dev/null
				then
					VAR=`top -b -n 2 -d 0.2 -p $pid | tail -1 | awk '{print $9}'`
					#VAR=`ps --pid $pid -o %cpu | tail -n 1 | sed 's/^ *//g'`
					THRESHOLD=10.0
					if (( $(echo "$VAR > $THRESHOLD" | bc -l) )) ; then processing=true; fi
				fi
			done
			if [ "$processing" = false ] ; then for pid in $pids; do kill -2 $pid; done; break; fi
		done

	rm -rf ${src_folder}${fname%.*} 

	local iter_end="$(date +%s)"
	echo -e "${job_prefix} Step took $((iter_end - iter_start)) secs"
}

files=$(ls -1 /data/ai_train_pcaps_merged); 
files_arr=($files)
declare -r files_count=${#files_arr[@]}

declare -r paralel_runs=3

task_start="$(date +%s)"
for (( x=0; x<${files_count}; x+=${paralel_runs} )); 
do

	paralel_run_pids=""
	ram_running=0
	for (( y=${x}; y<$((x+paralel_runs)); y+=1 ))
	do
		if (( y >= files_count)); then
			break
		fi

		target_pcap="${files_arr[${y}]}"
		prefix="[$((y+1))/${files_count}] - ${target_pcap} "
	
		if (( ram_running < 1 )); then
			ram_running+=1
			runnprobe "${target_pcap}" "${prefix}" "/data_hot/" &
			paralel_run_pids="${paralel_run_pids} $!"
		else
			runnprobe "${target_pcap}" "${prefix}" "/tmp/merging/" &
			paralel_run_pids="${paralel_run_pids} $!"
		fi
	done

	for p in $paralel_run_pids; do wait ${p}; done;
	echo -e "\n ${paralel_runs} ===========================================\n"

	# test 1 run
	# exit 1;
done

task_end="$(date +%s)"
echo -e "${job_prefix} task took $((task_end - task_start)) secs"

#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Use argument representing number of scenario!"
    exit 1
fi

x=$1

task_start="$(date +%s)"

paralel_run_pids=""

for ((day=19;day<=27;day+=1)); do
	python /data/kinit/generate_labels_denylist.py /data/kinit/flows/scenario_${x}/2022/08/${day}/ /data/kinit/attack_records_parseable & 
	paralel_run_pids="${paralel_run_pids} $!"
done

for p in $paralel_run_pids; do wait ${p}; done;
	
task_end="$(date +%s)"
echo -e "$task took $((task_end - task_start)) secs"

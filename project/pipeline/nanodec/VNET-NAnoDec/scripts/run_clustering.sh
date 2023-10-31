#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CLUSTERING_SRC_DIRPATH="$SCRIPT_DIRPATH"'/../src/clustering'


function echo_error_exit() {
  >&2 echo "$1"
  exit "${2:-1}"
}


function activate_conda_environment() {
  if ! type -f conda > /dev/null 2>&1; then
    echo_error_exit 'Error: "conda" is not in the PATH variable'
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$1"

  if [ $? -ne 0 ]; then
    echo_error_exit 'Error: conda environment '"$1"' does not exist. Make sure to create the environment by running setup.sh'
  fi
}


ip_lists_dirpath="$SCRIPT_DIRPATH"'/../ip_lists'
data_dirpath="$SCRIPT_DIRPATH"'/../data'
ip_filters_filepath="$SCRIPT_DIRPATH"'/../ip_filters/clusters_scenario_1_all_features.csv'
n_jobs=32
hyperloglog_error_rate=0.05
clustering_config_filepath="$CLUSTERING_SRC_DIRPATH"'/params_kmeans_production.json'


function usage() {
  echo 'Usage: '"${BASH_SOURCE[0]}"' [options] flows_dirpath'
  echo ''
  echo 'Assigns cluster labels to VNET IP addresses.'
  echo ''
  echo 'Positional arguments:'
  echo '* flows_dirpath: directory path containing network flows'
  echo ''
  echo 'Options:'
  echo '* --ip-lists: path to directory containing lists of VNET IP addresses with categories; default: '"$ip_lists_dirpath"
  echo '* --data-dir: directory path where features, models, logs and prediction results will be stored; default: '"$data_dirpath"
  echo '* --ip-filters: path to output file containing lists of VNET IP addresses with cluster labels; default: '"$ip_lists_dirpath"
  echo '* --n-jobs: number of parallel processes to run; default: '"$n_jobs"
  echo '* --hyperloglog-error-rate: error rate for HyperLogLog algorithm when computing unique tuple-related features; default: '"$hyperloglog_error_rate"
  echo '* --clustering-config: path to file containing clustering configuration (particularly hyperparameters and feature sets); default: '"$clustering_config_filepath"
}


parsed_args="$(getopt -o 'p:u:i:n:e:c:h' --long 'ip-lists:,data-dir:,ip-filters:,n-jobs:,hyperloglog-error-rate:,clustering-config:,help' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-p'|'--ip-lists')
      ip_lists_dirpath="$2"
      shift 2
      continue
    ;;
    '-u'|'--data-dir')
      data_dirpath="$2"
      shift 2
      continue
    ;;
    '-i'|'--ip-filters')
      ip_filters_filepath="$2"
      shift 2
      continue
    ;;
    '-n'|'--n-jobs')
      n_jobs="$2"
      shift 2
      continue
    ;;
    '-e'|'--hyperloglog-error-rate')
      hyperloglog_error_rate="$2"
      shift 2
      continue
    ;;
    '-c'|'--clustering-config')
      clustering_config_filepath="$2"
      shift 2
      continue
    ;;
    '-h'|'--help')
      >&2 usage
      exit 1
    ;;
    '--')
      shift
      break
    ;;
    *)
      break
    ;;
  esac
done

if [ $# -lt 1 ]; then
  >&2 usage
  exit 1
fi


flows_dirpath="$1"
shift


if [ ! -d "$flows_dirpath" ]; then
  echo_error_exit 'Path to flows "'"$flows_dirpath"'" does not exist'
fi


activate_conda_environment vnet

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Running clustering'

statistics_dirpath="$data_dirpath"'/clustering/statistics'
features_filepath="$data_dirpath"'/clustering/features.parquet'
clustering_results_filepath="$data_dirpath"'/clustering/results.pkl'

mkdir -p "$statistics_dirpath"

declare -a ip_lists_filepaths
while read -r filepath; do
  ip_lists_filepaths+=( "$filepath" )
done <<< "$(find "$ip_lists_dirpath" -type f)"


python "$CLUSTERING_SRC_DIRPATH"'/compute_statistics.py' \
  --force \
  --hyperloglog-error-rate "$hyperloglog_error_rate" \
  --n-jobs "$n_jobs" \
  "$flows_dirpath" \
  "$statistics_dirpath" \
  "${ip_lists_filepaths[@]}"


python "$CLUSTERING_SRC_DIRPATH"'/extract_features.py' \
  "$statistics_dirpath" \
  "$features_filepath"


python "$CLUSTERING_SRC_DIRPATH"'/cluster.py' \
  --n-jobs "$n_jobs" \
  --filter \
  "$features_filepath" \
  "$clustering_config_filepath" \
  "$clustering_results_filepath" \
  "${ip_lists_filepaths[@]}"


python "$CLUSTERING_SRC_DIRPATH"'/select_best_cluster_kmeans.py' \
  --filter \
  "$features_filepath" \
  "$clustering_results_filepath" \
  "$ip_filters_filepath"


echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

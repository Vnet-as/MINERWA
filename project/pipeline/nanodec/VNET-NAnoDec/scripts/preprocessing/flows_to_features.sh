#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PREPROCESSING_SCRIPTS_DIRPATH="$SCRIPT_DIRPATH"'/../../src/anomaly_detection/preprocessing'
SPLIT_SUFFIX='_split'
SHUFFLED_SUFFIX='_shuffled'
EXTRACTED_SUFFIX='_extracted'

INTERNAL_FILE_FORMAT='csv'


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


function exec_command_if_dir_is_empty() {
  local dir="$1"
  local message="$2"

  if dir_is_empty "$dir"; then
    echo "$message"
    "${@:3}"
  else
    echo 'Skipping: '"$message"': directory already exists'
  fi
}


function exec_command_if_file_does_not_exist() {
  local input_filepath="$1"
  local message="$2"

  if [ ! -f "$input_filepath" ]; then
    echo "$message"
    "${@:3}"
  else
    echo 'Skipping: '"$message"': file already exists'
  fi
}


function dir_is_empty() {
  if [ ! -d "$1" ] || [ -z "$(ls -A "$1")" ]; then
    return 0
  else
    return 1
  fi
}


input_file_format='gz'
output_file_format='gz'
windowing_config="$SCRIPT_DIRPATH"'/../../config/window_cfg.yaml'
ip_filters_filepath="$SCRIPT_DIRPATH"'/../../ip_filters/clusters_scenario_1_all_features.csv'
temp_dirpath='/data/kinit/temp'
remove_temp_data_tlag=''
n_jobs_split=32
n_jobs_flow_processing=32
n_jobs_shuffle=32
shuffle=0


parsed_args="$(getopt -o 'f:o:w:i:rhp:n:u:t:' --long 'input-file-format:,output-file-format:,windowing-config:,ip-filters:,remove-input,n-jobs-flow-processing:,n-jobs-split:,n-jobs-shuffle:,temp-dir:,shuffle' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-f'|'--input-file-format')
      input_file_format="$2"
      shift 2
      continue
    ;;
    '-o'|'--output-file-format')
      output_file_format="$2"
      shift 2
      continue
    ;;
    '-w'|'--windowing-config')
      windowing_config="$2"
      shift 2
      continue
    ;;
    '-i'|'--ip-filters')
      ip_filters_filepath="$2"
      shift 2
      continue
    ;;
    '-r'|'--remove-input')
      remove_temp_data_tlag='--remove-input'
      shift 1
      continue
    ;;
    '-h'|'--shuffle')
      shuffle=1
      shift 1
      continue
    ;;
    '-p'|'--n-jobs-flow-processing')
      n_jobs_flow_processing="$2"
      shift 2
      continue
    ;;
    '-n'|'--n-jobs-split')
      n_jobs_split="$2"
      shift 2
      continue
    ;;
    '-u'|'--n-jobs-shuffle')
      n_jobs_shuffle="$2"
      shift 2
      continue
    ;;
    '-t'|'--temp-dir')
      temp_dirpath="$2"
      shift 2
      continue
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

if [ $# -lt 2 ]; then
  echo_error_exit 'Usage: '"${BASH_SOURCE[0]}"' [options] flows_dirpath features_dirpath'
fi


flows_dirpath="$1"
shift
features_dirpath="$1"
shift


if [ ! -d "$flows_dirpath" ]; then
  echo_error_exit 'Path to flows "'"$flows_dirpath"'" does not exist'
fi

if [ ! -f "$windowing_config" ]; then
  echo_error_exit 'Windowing config does not exist at "'"$windowing_config"'"'
fi


activate_conda_environment vnet

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Processing flows in "'"$flows_dirpath"'"'


echo 'Extracting features'

features_dirpath_extracted="$features_dirpath""$EXTRACTED_SUFFIX"

mkdir -p "$features_dirpath_extracted"

python "$PREPROCESSING_SCRIPTS_DIRPATH"'/parallelproc.py' \
  --flows-dir "$flows_dirpath" \
  --flows-file-ext "$input_file_format" \
  --windowing-config "$windowing_config" \
  --n-jobs "$n_jobs_flow_processing" \
  --out-dir "$features_dirpath_extracted" \
  --remove-wincontexts \
  ${ip_filters_filepath:+--ip-filter "$ip_filters_filepath"} \
  --out-file-format "$INTERNAL_FILE_FORMAT" \
  --temp-dir "$temp_dirpath" \
  --do-not-merge


if [ $shuffle -eq 1 ]; then
  if [ "$INTERNAL_FILE_FORMAT" == 'csv' ]; then
    features_dirpath_split="$features_dirpath_extracted""$SPLIT_SUFFIX"

    echo 'Splitting flow files into smaller chunks in preparation for shuffling'
    
    "$PREPROCESSING_SCRIPTS_DIRPATH"'/csv_split.sh' \
      --n-jobs "$n_jobs_split" \
      $remove_temp_data_tlag \
      "$features_dirpath_extracted" \
      "$features_dirpath_split"
  else
    features_dirpath_split="$features_dirpath_extracted"
  fi
  
  features_dirpath_shuffled="$features_dirpath_split""$SHUFFLED_SUFFIX"
  
  echo 'Shuffling features'
  
  python "$PREPROCESSING_SCRIPTS_DIRPATH"'/shuffler.py' \
    --n-jobs "$n_jobs_shuffle" \
    --out-file-format "$output_file_format" \
    $remove_temp_data_tlag \
    "$features_dirpath_split" \
    "$features_dirpath_shuffled"
else
  features_dirpath_shuffled="$features_dirpath_extracted"
fi

mv -f "$features_dirpath_shuffled" "$features_dirpath"


echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

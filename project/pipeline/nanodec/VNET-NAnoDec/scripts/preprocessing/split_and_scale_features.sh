#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


TRAIN_SUFFIX='_train'
VALIDATION_SUFFIX='_valid'
TEST_SUFFIX='_test'
SCALED_SUFFIX='_scaled'
PER_CLUSTER_SUFFIX='_per_cluster'


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


function check_if_scaling_config_exists() {
  if [ "$processing_mode" == 'predict' ] && [ ! -f "$1" ]; then
    echo_error_exit 'Could not find scaling config in "'"$1"'". The scaling config must exist if processing_mode is "predict". Are you sure you ran this script with processing_mode equal to "evaluate" or "train"?'
  fi
}


function scale_data() {
  local scaling_conf_filepath="$1"
  local scripts_dirpath="$2"
  local train_dirpath="$3"
  local validation_dirpath="$4"
  local test_dirpath="$5"
  local scaling_type="$6"
  local n_jobs="$7"
  local train_scaled_dirpath="$8"
  local validation_scaled_dirpath="$9"
  local test_scaled_dirpath="${10}"
  
  if [ -z "$n_jobs" ]; then
    n_jobs=1
  fi
  
  if [ -z "$train_scaled_dirpath" ]; then
    train_scaled_dirpath="$train_dirpath""$SCALED_SUFFIX"
  fi
  
  if [ -z "$validation_scaled_dirpath" ]; then
    validation_scaled_dirpath="$validation_dirpath""$SCALED_SUFFIX"
  fi
  
  if [ -z "$test_scaled_dirpath" ]; then
    test_scaled_dirpath="$test_dirpath""$SCALED_SUFFIX"
  fi
  
  mkdir -p "$(dirname "$scaling_conf_filepath")"
  
  if [ "$processing_mode" == 'evaluate' ] || [ "$processing_mode" == 'train' ]; then
    exec_command_if_file_does_not_exist \
      "$scaling_conf_filepath" \
      'Creating scaling coefficients from the train set to '"$scaling_conf_filepath" \
      python "$scripts_dirpath"'/scaling_config_creator.py' "$train_dirpath" "$scaling_conf_filepath" --default-scaling-type "$scaling_type"

    exec_command_if_dir_is_empty \
      "$train_scaled_dirpath" \
      'Scaling train set, saving to '"$train_scaled_dirpath" \
      run_scaler "$train_dirpath" "$train_scaled_dirpath" "$scaling_conf_filepath" "$n_jobs"

    exec_command_if_dir_is_empty \
      "$validation_scaled_dirpath" \
      'Scaling validation set, saving to '"$validation_scaled_dirpath" \
      run_scaler "$validation_dirpath" "$validation_scaled_dirpath" "$scaling_conf_filepath" "$n_jobs"
  fi

  if [ "$processing_mode" == 'evaluate' ] || [ "$processing_mode" == 'predict' ]; then
    exec_command_if_dir_is_empty \
      "$test_scaled_dirpath" \
      'Scaling test set, saving to '"$test_scaled_dirpath" \
      run_scaler "$test_dirpath" "$test_scaled_dirpath" "$scaling_conf_filepath" "$n_jobs"
  fi
}


function run_scaler() {
  python "$scripts_dirpath"'/scaler.py' "$1" "$2" "$3" --scaling-max "$scaling_upper_bound_percentile" --n-jobs "$4"
}


function dir_is_empty() {
  if [ ! -d "$1" ] || [ -z "$(ls -A "$1")" ]; then
    return 0
  else
    return 1
  fi
}


features_dirpath=''
output_dirpath_suffix=''
scaling_conf_filepath=''
scaling_upper_bound_percentile=90
scaling_type='minmax'
scripts_dirpath="$SCRIPT_DIRPATH"'/../../src/anomaly_detection/preprocessing'
random_state=42
split_n_jobs='-1'
scale_n_jobs='-1'
ip_filters_filepath=''
processing_mode='train'


parsed_args="$(getopt -o 'f:s:p:t:r:n:j:' --long 'ip-filters:,output-dir-suffix:,scaling-upper-bound-percentile:,scaling-type:,random-state:,split-n-jobs:,scale-n-jobs:' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-f'|'--ip-filters')
      ip_filters_filepath="$2"
      shift 2
      continue
    ;;
    '-s'|'--output-dir-suffix')
      output_dirpath_suffix="$2"
      shift 2
      continue
    ;;
    '-p'|'--scaling-upper-bound-percentile')
      scaling_upper_bound_percentile="$2"
      shift 2
      continue
    ;;
    '-t'|'--scaling-type')
      scaling_type="$2"
      shift 2
      continue
    ;;
    '-r'|'--random-state')
      random_state="$2"
      shift 2
      continue
    ;;
    '-n'|'--split-n-jobs')
      split_n_jobs="$2"
      shift 2
      continue
    ;;
    '-j'|'--scale-n-jobs')
      scale_n_jobs="$2"
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
  echo_error_exit 'Usage: '"${BASH_SOURCE[0]}"' [options] processing_mode features_dirpath'
fi


processing_mode="$1"
shift
features_dirpath="$1"
shift

scaling_conf_filepath="$features_dirpath"'_scaling_configs/scaling_config'"$output_dirpath_suffix"'.yaml'


if [ "$processing_mode" != 'evaluate' ] && [ "$processing_mode" != 'train' ] && [ "$processing_mode" != 'predict' ]; then
  echo_error_exit 'processing_mode must be one of the following: "evaluate", "train" or "predict"'
fi

check_if_scaling_config_exists "$scaling_conf_filepath"


activate_conda_environment vnet

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Processing data in '"$features_dirpath"

# Splitting and scaling for the entire dataset

dataset_train_dirpath="$features_dirpath""$TRAIN_SUFFIX""$output_dirpath_suffix"
dataset_validation_dirpath="$features_dirpath""$VALIDATION_SUFFIX""$output_dirpath_suffix"
dataset_test_dirpath="$features_dirpath""$TEST_SUFFIX""$output_dirpath_suffix"

if [ "$processing_mode" == 'evaluate' ]; then
  exec_command_if_dir_is_empty \
    "$dataset_train_dirpath" \
    'Splitting files into train, validation and test sets' \
    python "$scripts_dirpath"'/data_splitter.py' \
      "$features_dirpath" \
      "$dataset_train_dirpath" \
      "$dataset_validation_dirpath" \
      --test-dirpath "$dataset_test_dirpath" \
      --random-state "$random_state" \
      --n-jobs "$split_n_jobs"
elif [ "$processing_mode" == 'train' ]; then
  exec_command_if_dir_is_empty \
    "$dataset_train_dirpath" \
    'Splitting files into train and validation sets' \
    python "$scripts_dirpath"'/data_splitter.py' \
      "$features_dirpath" \
      "$dataset_train_dirpath" \
      "$dataset_validation_dirpath" \
      --random-state "$random_state" \
      --n-jobs "$split_n_jobs"
elif [ "$processing_mode" == 'predict' ]; then
  exec_command_if_dir_is_empty \
    "$dataset_test_dirpath" \
    'Renaming "'"$features_dirpath"'" to "'"$dataset_test_dirpath"'"' \
    mv -f "$features_dirpath" "$dataset_test_dirpath"
fi

scale_data \
  "$scaling_conf_filepath" \
  "$scripts_dirpath" \
  "$dataset_train_dirpath" \
  "$dataset_validation_dirpath" \
  "$dataset_test_dirpath" \
  "$scaling_type" \
  "$scale_n_jobs"


# Splitting and scaling per cluster

if [ -f "$ip_filters_filepath" ]; then
  dataset_per_cluster_train_dirpath="$dataset_train_dirpath""$PER_CLUSTER_SUFFIX"
  dataset_per_cluster_validation_dirpath="$dataset_validation_dirpath""$PER_CLUSTER_SUFFIX"
  dataset_per_cluster_test_dirpath="$dataset_test_dirpath""$PER_CLUSTER_SUFFIX"
  
  echo 'Splitting files per cluster'
  
  if [ "$processing_mode" == 'evaluate' ] || [ "$processing_mode" == 'train' ]; then
    python "$scripts_dirpath"'/feature_splitter_by_clusters.py' "$dataset_train_dirpath" "$ip_filters_filepath" "$dataset_per_cluster_train_dirpath" --output-format 'gz'
    python "$scripts_dirpath"'/feature_splitter_by_clusters.py' "$dataset_validation_dirpath" "$ip_filters_filepath" "$dataset_per_cluster_validation_dirpath" --output-format 'gz'
  fi
  
  if [ "$processing_mode" == 'evaluate' ] || [ "$processing_mode" == 'predict' ]; then
    python "$scripts_dirpath"'/feature_splitter_by_clusters.py' "$dataset_test_dirpath" "$ip_filters_filepath" "$dataset_per_cluster_test_dirpath" --output-format 'gz'
  fi
  
  if [ "$processing_mode" == 'evaluate' ] || [ "$processing_mode" == 'train' ]; then
    cluster_dirpaths="$(find "$dataset_per_cluster_train_dirpath" -mindepth 1 -maxdepth 1 -type d)"
  elif [ "$processing_mode" == 'predict' ]; then
    cluster_dirpaths="$(find "$dataset_per_cluster_test_dirpath" -mindepth 1 -maxdepth 1 -type d)"
  fi
  
  while read -r cluster_dirpath; do
    cluster_dirname="$(basename "$cluster_dirpath")"
    scaling_conf_filepath_per_cluster="$features_dirpath"'_scaling_configs/scaling_config_'"$cluster_dirname""$output_dirpath_suffix"'.yaml'
    
    check_if_scaling_config_exists "$scaling_conf_filepath_per_cluster"
    
    train_dirpath_for_cluster="$dataset_per_cluster_train_dirpath"'/'"$cluster_dirname"
    validation_dirpath_for_cluster="$dataset_per_cluster_validation_dirpath"'/'"$cluster_dirname"
    test_dirpath_for_cluster="$dataset_per_cluster_test_dirpath"'/'"$cluster_dirname"
    
    train_dirpath_for_cluster_scaled="$dataset_per_cluster_train_dirpath""$SCALED_SUFFIX"'/'"$cluster_dirname"
    validation_dirpath_for_cluster_scaled="$dataset_per_cluster_validation_dirpath""$SCALED_SUFFIX"'/'"$cluster_dirname"
    test_dirpath_for_cluster_scaled="$dataset_per_cluster_test_dirpath""$SCALED_SUFFIX"'/'"$cluster_dirname"
    
    # Limit the number of parallel jobs per cluster since scaling per clsuter itself is run in parallel.
    scale_data \
      "$scaling_conf_filepath_per_cluster" \
      "$scripts_dirpath" \
      "$train_dirpath_for_cluster" \
      "$validation_dirpath_for_cluster" \
      "$test_dirpath_for_cluster" \
      "$scaling_type" \
      5 \
      "$train_dirpath_for_cluster_scaled" \
      "$validation_dirpath_for_cluster_scaled" \
      "$test_dirpath_for_cluster_scaled" \
      &
  done <<< "$cluster_dirpaths"
  
  wait
fi


echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

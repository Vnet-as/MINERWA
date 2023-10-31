#! /bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


function activate_conda_environment() {
  if ! type -f conda > /dev/null 2>&1; then
    echo 'Error: "conda" is not in the PATH variable'
    exit 1
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$1"

  if [ $? -ne 0 ]; then
    echo 'Error: conda environment '"$1"' does not exist. Make sure to create the environment by running setup.sh'
    exit 1
  fi
}


function dir_is_empty() {
  if [ ! -d "$1" ] || [ -z "$(ls -A "$1")" ]; then
    return 0
  else
    return 1
  fi
}


FOR_GEE_SUFFIX='_for_gee'

gpu=0
output_suffix=''
max_workers=30
scale_flag=''
add_labels_to_test_flag=''
add_labels_to_results_flag=''
spark_memory_usage_gb=50
spark_temp_dirpath='/data/kinit/temp/gee'
model_final_filename='final_model'
filter_model_filepath=''

enable_per_cluster=0
test_per_cluster_dirpath=''
models_per_cluster_dirpath=''
results_per_cluster_dirpath=''
logs_per_cluster_dirpath=''

parsed_args="$(getopt -o 'cp:d:r:l:af:g:o:w:sm:t:' --long 'enable-per-cluster,test-per-cluster:,models-per-cluster:,results-per-cluster:,logs-per-cluster:,add-labels-to-test,filter-model:,gpu:,max-workers:,scale,spark-memory-usage-gb:,spark-temp-dirpath:' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-c'|'--enable-per-cluster')
      enable_per_cluster=1
      shift 1
      continue
    ;;
    '-p'|'--test-per-cluster')
      test_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-d'|'--models-per-cluster')
      models_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-r'|'--results-per-cluster')
      results_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-l'|'--logs-per-cluster')
      logs_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-a'|'--add-labels-to-test')
      add_labels_to_test_flag='--add_labels_to_test'
      add_labels_to_results_flag='--add_labels_to_results'
      shift 1
      continue
    ;;
    '-f'|'--filter-model')
      filter_model_filepath="$2"
      shift 2
      continue
    ;;
    '-g'|'--gpu')
      gpu="$2"
      shift 2
      continue
    ;;
    '-w'|'--max-workers')
      random_state="$2"
      shift 2
      continue
    ;;
    '-s'|'--scale')
      scale_flag='--scale'
      shift
      continue
    ;;
    '-m'|'--spark-memory-usage-gb')
      spark_memory_usage_gb="$2"
      shift 2
      continue
    ;;
    '-t'|'--spark-temp-dirpath')
      spark_temp_dirpath="$2"
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

if [ $# -lt 3 ]; then
  echo 'Usage: '"${BASH_SOURCE[0]}"' [options] test_dirpath model_dirpath results_dirpath'
  exit 1
fi


test_dirpath="$1"
shift
model_dirpath="$1"
shift
results_dirpath="$1"
shift


if [[ $enable_per_cluster -eq 1 && ( -z "$test_per_cluster_dirpath" || -z "$models_per_cluster_dirpath" || -z "$results_per_cluster_dirpath" ) ]]; then
  echo 'If --enable-per-cluster is specified, you must also specify all other options ending with "-per-cluster"'
  exit 1
fi


function run_pipeline() {
  local test_dirpath="$1"
  local model_dirpath="$2"
  local results_dirpath="$3"
  local test_dirpath_for_model="$4"
  
  if [ -z "$test_dirpath_for_model" ]; then
    test_dirpath_for_model="$test_dirpath""$FOR_GEE_SUFFIX"
  fi

  mkdir -p "$spark_temp_dirpath"
  mkdir -p "$test_dirpath_for_model"

  echo 'Converting data to Petastorm format'

  if dir_is_empty "$test_dirpath_for_model"; then
    python "$SCRIPT_DIRPATH"'/build_model_input_vnet.py' \
      --test "$test_dirpath" \
      --target_test "$test_dirpath_for_model" \
      $scale_flag \
      $add_labels_to_test_flag \
      --clip \
      --spark_config 'spark.driver.memory' "$spark_memory_usage_gb"'g' \
      --spark_config 'spark.executor.memory' "$spark_memory_usage_gb"'g' \
      --spark_config 'spark.local.dir' "$spark_temp_dirpath"
  else
    echo 'Skipping conversion to Petastorm (already done)'
  fi


  echo 'Performing prediction'

  output_path="$results_dirpath"'_'"$(date '+%Y-%m-%d_%H-%M-%S')"
  
  mkdir -p "$output_path"
  
  if dir_is_empty "$output_path"; then
    python "$SCRIPT_DIRPATH"'/evaluate_vae.py' \
      --data_path "$test_dirpath_for_model" \
      --model_path "$model_dirpath"'/'"$model_final_filename" \
      --filter_model_path "$filter_model_filepath" \
      --output_path "$output_path" \
      --dataset 'vnet' \
      --gpu "$gpu" \
      --max_workers "$max_workers" \
      --add_ip_to_results \
      $add_labels_to_results_flag
  else
    echo 'Skipping evaluation - results already exist in '"$output_path"
  fi
}


activate_conda_environment vnet-gee

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Running GEE anomaly detection'

run_pipeline "$test_dirpath" "$model_dirpath" "$results_dirpath"

if [ $enable_per_cluster -eq 1 ]; then
  cluster_dirpaths="$(find "$test_per_cluster_dirpath" -mindepth 1 -maxdepth 1 -type d)"
  
  while read -r cluster_dirpath; do
    cluster_name="$(basename "$cluster_dirpath")"
    
    mkdir -p "$logs_per_cluster_dirpath"'/'"$cluster_name"
    
    run_pipeline \
      "$test_per_cluster_dirpath"'/'"$cluster_name" \
      "$models_per_cluster_dirpath"'/'"$cluster_name" \
      "$results_per_cluster_dirpath"'/'"$cluster_name" \
      "$test_per_cluster_dirpath""$FOR_GEE_SUFFIX"'/'"$cluster_name" \
      > "$logs_per_cluster_dirpath"'/'"$cluster_name"'/stdout.log' \
      2> "$logs_per_cluster_dirpath"'/'"$cluster_name"'/stderr.log' \
      &
  done <<< "$cluster_dirpaths"
  
  wait
fi

echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

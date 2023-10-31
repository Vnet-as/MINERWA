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
random_state=42
max_epochs=12
max_workers=30
scale_flag=''
spark_memory_usage_gb=50
spark_temp_dirpath='/data/kinit/temp/gee'
model_final_filename='final_model'

enable_per_cluster=0
train_per_cluster_dirpath=''
validation_per_cluster_dirpath=''
models_per_cluster_dirpath=''
logs_per_cluster_dirpath=''

parsed_args="$(getopt -o 'ca:v:d:l:g:o:r:e:w:sm:t:' --long 'enable-per-cluster,train-per-cluster:,validation-per-cluster:,models-per-cluster:,logs-per-cluster:,gpu:,random-state:,max-epochs:,max-workers:,scale,spark-memory-usage-gb:,spark-temp-dirpath:' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-c'|'--enable-per-cluster')
      enable_per_cluster=1
      shift 1
      continue
    ;;
    '-a'|'--train-per-cluster')
      train_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-v'|'--validation-per-cluster')
      validation_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-d'|'--models-per-cluster')
      models_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-l'|'--logs-per-cluster')
      logs_per_cluster_dirpath="$2"
      shift 2
      continue
    ;;
    '-g'|'--gpu')
      gpu="$2"
      shift 2
      continue
    ;;
    '-r'|'--random-state')
      random_state="$2"
      shift 2
      continue
    ;;
    '-e'|'--max-epochs')
      max_epochs="$2"
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
  echo 'Usage: '"${BASH_SOURCE[0]}"' [options] train_dirpath validation_dirpath model_dirpath'
  exit 1
fi


train_dirpath="$1"
shift
validation_dirpath="$1"
shift
model_dirpath="$1"
shift


if [[ $enable_per_cluster -eq 1 && ( -z "$train_per_cluster_dirpath" || -z "$validation_per_cluster_dirpath" || -z "$models_per_cluster_dirpath" ) ]]; then
  echo 'If --enable-per-cluster is specified, you must also specify all other options ending with "-per-cluster"'
  exit 1
fi


function run_pipeline() {
  local train_dirpath="$1"
  local validation_dirpath="$2"
  local model_dirpath="$3"
  local train_dirpath_for_model="$4"
  local validation_dirpath_for_model="$5"
  
  if [ -z "$train_dirpath_for_model" ]; then
    train_dirpath_for_model="$train_dirpath""$FOR_GEE_SUFFIX"
  fi
  
  if [ -z "$validation_dirpath_for_model" ]; then
    validation_dirpath_for_model="$validation_dirpath""$FOR_GEE_SUFFIX"
  fi

  mkdir -p "$spark_temp_dirpath"
  mkdir -p "$train_dirpath_for_model"
  mkdir -p "$validation_dirpath_for_model"

  echo 'Converting data to Petastorm format'

  if dir_is_empty "$train_dirpath_for_model"; then
    python "$SCRIPT_DIRPATH"'/build_model_input_vnet.py' \
      --train "$train_dirpath" \
      --validate "$validation_dirpath" \
      --target_train "$train_dirpath_for_model" \
      --target_validate "$validation_dirpath_for_model" \
      $scale_flag \
      --clip \
      --spark_config 'spark.driver.memory' "$spark_memory_usage_gb"'g' \
      --spark_config 'spark.executor.memory' "$spark_memory_usage_gb"'g' \
      --spark_config 'spark.local.dir' "$spark_temp_dirpath"
  else
    echo 'Skipping conversion to Petastorm (already done)'
  fi


  echo 'Training model'

  mkdir -p "$model_dirpath"

  if [ ! -f "$model_dirpath"'/'"$model_final_filename" ]; then
    python "$SCRIPT_DIRPATH"'/train_vae.py' \
      --data_path "$train_dirpath_for_model" \
      --validation_path "$validation_dirpath_for_model" \
      --model_path "$model_dirpath" \
      --final-model-filename "$model_final_filename" \
      --dataset 'vnet' \
      --gpu "$gpu" \
      --max_epochs "$max_epochs" \
      --max_workers "$max_workers" \
      --log_loss \
      --random_state "$random_state" \
      --deterministic
  else
    echo 'Skipping training - model already exists'
  fi
}


activate_conda_environment vnet-gee

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Training GEE-based anomaly detector'

run_pipeline "$train_dirpath" "$validation_dirpath" "$model_dirpath"

if [ $enable_per_cluster -eq 1 ]; then
  cluster_dirpaths="$(find "$train_per_cluster_dirpath" -mindepth 1 -maxdepth 1 -type d)"
  
  while read -r cluster_dirpath; do
    cluster_name="$(basename "$cluster_dirpath")"
    
    mkdir -p "$logs_per_cluster_dirpath"'/'"$cluster_name"
    
    run_pipeline \
      "$train_per_cluster_dirpath"'/'"$cluster_name" \
      "$validation_per_cluster_dirpath"'/'"$cluster_name" \
      "$models_per_cluster_dirpath"'/'"$cluster_name" \
      "$train_per_cluster_dirpath""$FOR_GEE_SUFFIX"'/'"$cluster_name" \
      "$validation_per_cluster_dirpath""$FOR_GEE_SUFFIX"'/'"$cluster_name" \
      > "$logs_per_cluster_dirpath"'/'"$cluster_name"'/stdout.log' \
      2> "$logs_per_cluster_dirpath"'/'"$cluster_name"'/stderr.log' \
      &
  done <<< "$cluster_dirpaths"
  
  wait
fi

echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

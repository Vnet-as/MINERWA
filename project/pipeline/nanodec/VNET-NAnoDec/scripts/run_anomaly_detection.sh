#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PREPROCESSING_SCRIPTS_DIRPATH="$SCRIPT_DIRPATH"'/preprocessing'
GEE_DIRPATH="$SCRIPT_DIRPATH"'/../../VNET-NAnoDec-GEE'


function echo_error_exit() {
  >&2 echo "$1"
  exit "${2:-1}"
}


input_file_format='gz'
features_file_format='gz'
data_dirpath="$SCRIPT_DIRPATH"'/../data'
ip_filters_filepath="$SCRIPT_DIRPATH"'/../ip_filters/clusters_scenario_1_all_features.csv'
shuffle=0
n_jobs_flows_to_features=32
temp_dirpath='/data/kinit/temp'
random_state_evaluate=42
gpu=0
filter_model_filepath=''


function usage() {
  echo 'Usage: '"${BASH_SOURCE[0]}"' [options] flows_dirpath mode'
  echo ''
  echo 'Runs anomaly detection pipeline using network flows as input.'
  echo ''
  echo 'Positional arguments:'
  echo '* flows_dirpath: directory path containing network flows'
  echo '* mode: processing mode; can have one of the following values:'
  echo '  * train - train anomaly detection model'
  echo '  * predict - perform prediction using existing anomaly detection model'
  echo '  * evaluate - train anomaly detection model and perform prediction using this model. This is useful for experimenting to fine-tune the model detection accuracy.'
  echo ''
  echo 'Options:'
  echo '* --input-file-format: file format and extension of the input network flows; default: '"$input_file_format"
  echo '* --features-file-format: file format and extension of extracted features used for training/prediction. Applies to "train" and "evaluate" modes only. For the "predict" mode, the file format is always CSV; default: '"$features_file_format"
  echo '* --data-dir: directory path where features, models, logs and prediction results will be stored; default: '"$data_dirpath"
  echo '* --ip-filters: Path to file containing IP addresses used for filtering flows (applies only if mode is not "predict") and splitting features to clusters (applies for all modes); default: '"$ip_filters_filepath"
  echo '* --shuffle: if specified, shuffle extracted features randomly'
  echo '* --n-jobs-flows-to-features: number of concurrent jobs for feature extraction-related scripts; default: '"$n_jobs_flows_to_features"
  echo '* --temp-dir: directory path storing temporary data; default: '"$temp_dirpath"
  echo '* --random-state-evaluate: fixed random state to use if the mode is "evaluate"; default: '"$random_state_evaluate"
  echo '* --gpu: GPU ID to use for model training; default: '"$gpu"
  echo '* --filter-model: File path to a classifier model used for filtering during anomaly detection if the mode is "evaluate" or "predict"; default: '"$filter_model_filepath"
}


parsed_args="$(getopt -o 'f:o:u:i:en:t:d:g:l:h' --long 'input-file-format:,features-file-format:,data-dir:,ip-filters:,shuffle,n-jobs-flows-to-features:,temp-dir:,random-state-evaluate:,gpu:,filter-model:,help' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-f'|'--input-file-format')
      input_file_format="$2"
      shift 2
      continue
    ;;
    '-o'|'--features-file-format')
      features_file_format="$2"
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
    '-e'|'--shuffle')
      shuffle=1
      shift 1
      continue
    ;;
    '-n'|'--n-jobs-flows-to-features')
      n_jobs_flows_to_features="$2"
      shift 2
      continue
    ;;
    '-t'|'--temp-dir')
      temp_dirpath="$2"
      shift 2
      continue
    ;;
    '-d'|'--random-state-evaluate')
      random_state_evaluate="$2"
      shift 2
      continue
    ;;
    '-g'|'--gpu')
      gpu="$2"
      shift 2
      continue
    ;;
    '-l'|'--filter-model')
      filter_model_filepath="$2"
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

if [ $# -lt 2 ]; then
  >&2 usage
  exit 1
fi


flows_dirpath="$1"
shift
mode="$1"
shift


if [ ! -d "$flows_dirpath" ]; then
  echo_error_exit 'Path to flows "'"$flows_dirpath"'" does not exist'
fi

if [ "$mode" != 'train' ] && [ "$mode" != 'predict' ] && [ "$mode" != 'evaluate' ]; then
  echo_error_exit 'mode must be one of: "train", "predict" or "evaluate"'
fi


echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Running anomaly detection pipeline with mode "'"$mode"'"'

logs_current_run_dirpath="$data_dirpath"'/logs/'"$(date '+%Y-%m-%d_%H-%M-%S')"
features_dirpath="$data_dirpath"'/features'
models_dirpath="$data_dirpath"'/models'
results_dirpath="$data_dirpath"'/results'

mkdir -p "$logs_current_run_dirpath"
mkdir -p "$data_dirpath"
mkdir -p "$features_dirpath"
mkdir -p "$models_dirpath"


if [ "$mode" == 'train' ]; then
  # For production environments, it is generally not recommended to use a fixed random state.
  random_state=-1
  ip_filters_filepath_for_feature_extraction="$ip_filters_filepath"
  shuffle_flag='--shuffle'
elif [ "$mode" == 'predict' ]; then
  # For production environments, it is generally not recommended to use a fixed random state.
  random_state=-1
  # Flows with unknown IP addresses must not be filtered during prediction.
  # The generic model will be used for anomaly detection for such flows.
  ip_filters_filepath_for_feature_extraction=''
  # Shuffling is unnecessary for prediction.
  shuffle_flag=''
elif [ "$mode" == 'evaluate' ]; then
  random_state="$random_state_evaluate"
  ip_filters_filepath_for_feature_extraction="$ip_filters_filepath"
  shuffle_flag='--shuffle'
fi


echo 'Transforming flows to features'

"$PREPROCESSING_SCRIPTS_DIRPATH"'/flows_to_features.sh' \
  $shuffle_flag \
  --remove-input \
  ${ip_filters_filepath_for_feature_extraction:+--ip-filters "$ip_filters_filepath_for_feature_extraction"} \
  --input-file-format "$input_file_format" \
  --output-file-format "$features_file_format" \
  --n-jobs-flow-processing "$n_jobs_flows_to_features" \
  --n-jobs-split "$n_jobs_flows_to_features" \
  --n-jobs-shuffle "$n_jobs_flows_to_features" \
  --temp-dir "$temp_dirpath" \
  "$flows_dirpath" \
  "$features_dirpath" \
  > "$logs_current_run_dirpath"'/flows_to_features_'"$mode"'_stdout.log' \
  2> "$logs_current_run_dirpath"'/flows_to_features_'"$mode"'_stderr.log'


echo 'Processing features before being fed into the anomaly detection model'

"$PREPROCESSING_SCRIPTS_DIRPATH"'/split_and_scale_features.sh' \
  --scaling-type minmax \
  --random-state "$random_state" \
  --ip-filters "$ip_filters_filepath" \
  "$mode" \
  "$features_dirpath" \
  > "$logs_current_run_dirpath"'/split_and_scale_features_'"$mode"'_stdout.log' \
  2> "$logs_current_run_dirpath"'/split_and_scale_features_'"$mode"'_stderr.log'


if [ "$mode" == 'train' ]; then
  echo 'Training anomaly detection model'
  
  "$GEE_DIRPATH"'/run_pipeline_vae_vnet_train.sh' \
    --enable-per-cluster \
    --train-per-cluster "$features_dirpath"'_train_per_cluster_scaled' \
    --validation-per-cluster "$features_dirpath"'_valid_per_cluster_scaled' \
    --models-per-cluster "$models_dirpath"'/per_cluster' \
    --logs-per-cluster "$logs_current_run_dirpath"'/models_per_cluster_'"$mode" \
    --gpu "$gpu" \
    --random-state "$random_state" \
    --spark-temp-dirpath "$temp_dirpath"'/gee' \
    "$features_dirpath"'_train_scaled' \
    "$features_dirpath"'_valid_scaled' \
    "$models_dirpath"'/generic' \
    > "$logs_current_run_dirpath"'/generic_model_'"$mode"'_stdout.log' \
    2> "$logs_current_run_dirpath"'/generic_model_'"$mode"'_stderr.log'
elif [ "$mode" == 'evaluate' ]; then
  echo 'Training and evaluating anomaly detection model'
  
  "$GEE_DIRPATH"'/run_pipeline_vae_vnet_train.sh' \
    --enable-per-cluster \
    --train-per-cluster "$features_dirpath"'_train_per_cluster_scaled' \
    --validation-per-cluster "$features_dirpath"'_valid_per_cluster_scaled' \
    --models-per-cluster "$models_dirpath"'/per_cluster' \
    --logs-per-cluster "$logs_current_run_dirpath"'/models_per_cluster_train' \
    --gpu "$gpu" \
    --random-state "$random_state" \
    --spark-temp-dirpath "$temp_dirpath"'/gee' \
    "$features_dirpath"'_train_scaled' \
    "$features_dirpath"'_valid_scaled' \
    "$models_dirpath"'/generic' \
    > "$logs_current_run_dirpath"'/generic_model_train_stdout.log' \
    2> "$logs_current_run_dirpath"'/generic_model_train_stderr.log'
  
  "$GEE_DIRPATH"'/run_pipeline_vae_vnet_predict.sh' \
    --enable-per-cluster \
    --test-per-cluster "$features_dirpath"'_test_per_cluster_scaled' \
    --models-per-cluster "$models_dirpath"'/per_cluster' \
    --results-per-cluster "$results_dirpath"'/per_cluster' \
    --logs-per-cluster "$logs_current_run_dirpath"'/models_per_cluster_predict' \
    --gpu "$gpu" \
    --add-labels-to-test \
    --spark-temp-dirpath "$temp_dirpath"'/gee' \
    ${filter_model_filepath:+--filter-model "$filter_model_filepath"} \
    "$features_dirpath"'_test_scaled' \
    "$models_dirpath"'/generic' \
    "$results_dirpath"'/generic' \
    > "$logs_current_run_dirpath"'/generic_model_predict_stdout.log' \
    2> "$logs_current_run_dirpath"'/generic_model_predict_stderr.log'
elif [ "$mode" == 'predict' ]; then
  echo 'Performing prediction using the trained anomaly detection model'
  
  "$GEE_DIRPATH"'/run_pipeline_vae_vnet_predict.sh' \
    --enable-per-cluster \
    --test-per-cluster "$features_dirpath"'_test_per_cluster_scaled' \
    --models-per-cluster "$models_dirpath"'/per_cluster' \
    --results-per-cluster "$results_dirpath"'/per_cluster' \
    --logs-per-cluster "$logs_current_run_dirpath"'/models_per_cluster_'"$mode" \
    --gpu "$gpu" \
    --spark-temp-dirpath "$temp_dirpath"'/gee' \
    ${filter_model_filepath:+--filter-model "$filter_model_filepath"} \
    "$features_dirpath"'_test_scaled' \
    "$models_dirpath"'/generic' \
    "$results_dirpath"'/generic' \
    > "$logs_current_run_dirpath"'/generic_model_'"$mode"'_stdout.log' \
    2> "$logs_current_run_dirpath"'/generic_model_'"$mode"'_stderr.log'
fi


echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

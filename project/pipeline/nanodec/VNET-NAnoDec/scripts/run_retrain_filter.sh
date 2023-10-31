#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FILTER_SRC_DIRPATH="$SCRIPT_DIRPATH"'/../src/filter'


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


n_jobs=32
temp_dirpath='/data/kinit/temp'
extracted_attacks_filename='extracted_attacks.csv.gz'
extracted_attacks_balanced_filename='extracted_attacks_binary-balanced.csv.gz'


function usage() {
  echo 'Usage: '"${BASH_SOURCE[0]}"' [options] flows_dirpath filter_model_filepath'
  echo ''
  echo 'Trains a classification-based filter of known network attacks from network flows.'
  echo ''
  echo 'Positional arguments:'
  echo '* flows_dirpath: directory path containing network flows'
  echo '* filter_model_filepath: file path of the output model'
  echo ''
  echo 'Options:'
  echo '* --n-jobs: number of parallel jobs to run when training the filter; default: '"$n_jobs"
  echo '* --temp-dir: directory path storing temporary data; default: '"$temp_dirpath"
}


parsed_args="$(getopt -o 'n:t:h' --long 'n-jobs:,temp-dir:,help' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-n'|'--n-jobs')
      n_jobs="$2"
      shift 2
      continue
    ;;
    '-t'|'--temp-dir')
      temp_dirpath="$2"
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
filter_model_filepath="$1"
shift


if [ ! -d "$flows_dirpath" ]; then
  echo_error_exit 'Path to flows "'"$flows_dirpath"'" does not exist'
fi


activate_conda_environment vnet

echo 'Start date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Training classification-based filter'

mkdir -p "$temp_dirpath"


python "$FILTER_SRC_DIRPATH"'/extract_attacks.py' \
  "$flows_dirpath" \
  "$temp_dirpath"'/'"$extracted_attacks_filename"

python "$FILTER_SRC_DIRPATH"'/binary_balance_data.py' \
  "$temp_dirpath"'/'"$extracted_attacks_filename" \
  "$temp_dirpath"'/'"$extracted_attacks_balanced_filename"

python "$FILTER_SRC_DIRPATH"'/train_rfc_filter.py' \
  --n-jobs "$n_jobs" \
  "$temp_dirpath"'/'"$extracted_attacks_balanced_filename" \
  "$filter_model_filepath"

rm "$temp_dirpath"'/'"$extracted_attacks_filename"
rm "$temp_dirpath"'/'"$extracted_attacks_balanced_filename"


echo 'End date: '"$(date '+%Y-%m-%d_%H-%M-%S')"
echo 'Done'

conda deactivate

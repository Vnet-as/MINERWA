#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


function print_usage() {
  echo 'Splits a single CSV file or a directory containing CSV files into multiple files in a new directory by preserving header, removing the old file(s) if desired.'
  echo ''
  echo "Usage: ${BASH_SOURCE[0]} input_path output_dirpath"
  echo ''
  echo 'Options:'
  echo '* --remove-input: if specified, remove input files'
  echo '* --chunk-size: number of rows per split file'
  echo '* --n-jobs: number of processes to run in parallel'
}


function dir_exists_and_is_empty() {
  if [ -d "$1" ] && [ -z "$(ls -A "$1")" ]; then
    return 0
  else
    return 1
  fi
}


function split_csv() {
  local filepath="$1"
  local output_dirpath="$2"
  local chunk_size="$3"
  local remove_input="$4"
  
  local file_basename_root="$(basename "${filepath%.*}")"
  local filepath_ext="${filepath##*.}"
  local new_dirpath="$output_dirpath"'/'"$file_basename_root"
  local header="$(head -n 1 "$filepath")"

  # Create output directory and split the text
  mkdir -p "$new_dirpath"
  tail -n +2 "$filepath" | split -l "$chunk_size" - "${new_dirpath}/${file_basename_root}_"

  # Append the CSV header to each split file
  while read -r split_filepath; do
    # FIXME: Shouldn't "$header" be escaped here?
    sed -i -e "1i$header" "$split_filepath"
    mv -f "$split_filepath" "$split_filepath"'.'"$filepath_ext"
  done <<< "$(find "$new_dirpath" -type f)"

  # Remove the source file if desired
  if [ $remove_input -eq 1 ]; then
    rm "$filepath"
  fi
}


chunk_size=10000
remove_input=0
n_jobs=32

parsed_args="$(getopt -o 'rs:n:h' --long 'remove-input,chunk-size:,n-jobs:,help' -n "${BASH_SOURCE[0]}" -- "$@")"

while true; do
  case "$1" in
    '-r'|'--remove-input')
      remove_input=1
      shift 1
      continue
    ;;
    '-s'|'--chunk-size')
      chunk_size="$2"
      shift 2
      continue
    ;;
    '-n'|'--n-jobs')
      n_jobs="$2"
      shift 2
      continue
    ;;
    '-h'|'--help')
      >&2 print_usage
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
  >&2 print_usage
  exit 1
fi


input_path="$1"
shift
output_dirpath="$1"
shift


if [ -f "$input_path" ]; then
  split_csv "$1"
elif [ -d "$input_path" ]; then
  export -f split_csv
  SHELL="$(type -p bash)" find "$input_path" -type f | parallel split_csv {} "$output_dirpath" "$chunk_size" "$remove_input"
  
  if [ $remove_input -eq 1 ] && dir_exists_and_is_empty "$input_path"; then
    rmdir "$input_path"
  fi
else
  >&2 echo 'Path "'"$input_path"'" does not exist'
  >&2 print_usage
  exit 1
fi

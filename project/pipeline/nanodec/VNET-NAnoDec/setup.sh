#!/bin/bash

SCRIPT_DIRPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

GEE_DIRPATH="$SCRIPT_DIRPATH"'/../VNET-NAnoDec-GEE'


if [ ! -d "$GEE_DIRPATH" ]; then
  >&2 echo 'Error: Could not find "'"$GEE_DIRPATH"'". Make sure the "VNET-NAnoDec-GEE" repository is located on the same directory level as "VNET-NAnoDec".'
fi


echo 'Installing packages'

apt-get install parallel


conda update --name base conda

echo 'Creating conda environments'

conda env create --file "$SCRIPT_DIRPATH"'/environment.yml'

conda env create --file "$GEE_DIRPATH"'/environment.yml'

echo 'Done'

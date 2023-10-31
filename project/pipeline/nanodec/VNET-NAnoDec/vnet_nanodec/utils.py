"""
Utility functions used in other modules.
"""

import gzip
import importlib
import os
from types import ModuleType
from typing import Any, Optional, Tuple, Union
import warnings

import dask.dataframe as dd
import pandas as pd
import pyarrow


DEFAULT_CSV_SEPARATOR_LOAD = '|'
DEFAULT_CSV_SEPARATOR_SAVE = ','
VALID_OUTPUT_FILE_FORMATS = ['csv', 'feather', 'parquet', 'arrow', 'pickle', 'gz']


def load_df(
        filepath: str,
        csv_sep: Optional[str] = DEFAULT_CSV_SEPARATOR_LOAD,
        dtypes: Optional[dict] = None,
        warn_on_unknown: bool = True,
        use_dask: bool = False,
        **kwargs,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Loads a DataFrame from a specified file path.  Automatically detect the file type based on
    its extension and calls an appropriate loading function.

    Parameters:
        filepath -- path to the DataFrame to load
        csv_sep  -- CSV separator.  If None, performs its automatic detection
        dypes    -- dictionary specifiying data types of the loaded DataFrame
        warn_on_unknown -- issues a warning if unrecognized file extension is used
        use_dask -- if True, use dask.dataframe to read file(s) instead of pandas. Applies to 'csv',
                    'gz' and 'parquet' formats only.
        **kwargs -- Additional keyword arguments specific to a file format. Use this only if you are
                    sure which file format should be used for loading.
    Returns:
        pd.DataFrame -- Pandas DataFrame object of the loaded DataFrame from the disk"""
    dtypes_already_set = False

    if filepath.endswith('.gz'):
        if not use_dask:
            with gzip.open(filepath, 'rb') as f:
                df = pd.read_csv(f, sep=csv_sep, dtype=dtypes, **kwargs)
        else:
            df = dd.read_csv(filepath, sep=csv_sep, dtype=dtypes, compression='gzip', **kwargs)

        dtypes_already_set = True
    elif filepath.endswith('.parquet'):
        if not use_dask:
            df = pd.read_parquet(filepath, **kwargs)
        else:
            df = dd.read_parquet(filepath, **kwargs)
    elif filepath.endswith('.feather'):
        df = pd.read_feather(filepath, **kwargs)
    elif filepath.endswith('.pkl'):
        df = pd.read_pickle(filepath, compression='zip', **kwargs)
    elif filepath.endswith('.pkl-zip'):
        df = pd.read_pickle(filepath, compression='zip', **kwargs)
    elif filepath.endswith(('.csv', '.flows')):
        if not use_dask:
            df = pd.read_csv(filepath, sep=csv_sep, dtype=dtypes, **kwargs)
        else:
            df = dd.read_csv(filepath, sep=csv_sep, dtype=dtypes, **kwargs)

        dtypes_already_set = True
    else:
        if warn_on_unknown:
            warnings.warn('Unrecognized file extension, attempting loading via pandas.read_pickle')
        df = pd.read_pickle(filepath, **kwargs)

    if dtypes and not dtypes_already_set:
        df = df.astype(dtypes)

    return df


def save_df(
        df: pd.DataFrame, filepath: str, file_format: Optional[str] = None,
        csv_sep: Optional[str] = DEFAULT_CSV_SEPARATOR_SAVE) -> str:
    """Saves a dataframe with a specified format to disk.

    Parameters:
        df          -- DataFrame to save
        filepath    -- Filepath to save file to. If file_format is None, this is a full file path.
                       If file_format is not None, this must be a file path without file extension.
        file_format -- Format to use for saving. If None, file format is inferred from filepath.
        csv_sep     -- CSV separator to use

    Returns:
        str -- Filepath of the created file"""

    if file_format is not None:
        filepath_final = filepath
    else:
        filepath_final, file_ext = os.path.splitext(filepath)
        file_format = file_ext[1:]

    if file_format == 'csv' or file_format == 'flows':
        filepath_final += '.' + file_format
        df.to_csv(filepath_final, index=False, sep=csv_sep)
    elif file_format == 'gz':
        filepath_final += '.gz'
        with gzip.open(filepath_final, 'wb') as f:
            df.to_csv(f, index=False, sep=csv_sep)
    elif file_format == 'feather':
        filepath_final += '.feather'
        df.to_feather(filepath_final)
    elif file_format == 'parquet':
        filepath_final += '.parquet'
        df.to_parquet(filepath_final)
    elif file_format == 'arrow':
        filepath_final += '.arrow.gz'
        with pyarrow.CompressedOutputStream(filepath_final, 'gzip') as out:
            pyarrow.csv.write_csv(pyarrow.Table.from_pandas(df, preserve_index=False), out)
    elif file_format == 'pickle':
        filepath_final += '.pkl-zip'
        df.to_pickle(filepath_final, compression='zip')
    else:
        raise ValueError(f'Invalid output file format; valid values: {VALID_OUTPUT_FILE_FORMATS}')

    return filepath_final


def load_module_and_obj(obj_full_name: str) -> Tuple[ModuleType, Any]:
    """Dynamically loads an object and its corresponding module given the full object path."""
    module_path, _, obj_name = obj_full_name.rpartition('.')
    module = importlib.import_module(module_path)

    return module, getattr(module, obj_name)


def seconds_to_milliseconds(seconds: int) -> int:
    return int(seconds * 1000)


def milliseconds_to_seconds(milliseconds: int) -> float:
    return milliseconds / 1000.0


def microseconds_to_milliseconds(microseconds: int) -> float:
    return microseconds * 0.001


def microseconds_to_seconds(microseconds: int) -> float:
    return microseconds * 0.000001

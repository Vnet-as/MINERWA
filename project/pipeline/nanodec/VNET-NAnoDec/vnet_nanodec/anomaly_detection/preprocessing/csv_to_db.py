"""Converts CSV files in a given folder to a CSV database

Usage:
    python csv_to_db.py in_dirpath out_dbpath table_name
"""

import os
import pandas as pd
import sqlite3
import sys


FILEEXT = 'csv'


def main(args: list) -> None:
    if len(args) != 4:
        raise Exception("Usage: {} in_dirpath out_dbpath table_name".format(args[0]))

    # Map parameters to variables for better readability
    input_dirpath       = args[1]
    output_db_filename  = args[2]
    output_db_tablename = args[3]

    # Obtain files of interest
    csv_files = [filename for filename in os.listdir(input_dirpath) if
        filename.endswith(FILEEXT)]

    # Create SQLLite database connection
    sql_con = sqlite3.connect(output_db_filename)

    # Convert CSVs into database by appending
    for csv_file in csv_files:
        data_df = pd.read_csv(os.path.join(input_dirpath, csv_file))

        data_df.to_sql(output_db_tablename, sql_con, if_exists='append', index=False)

if __name__ == '__main__':
    main(sys.argv)

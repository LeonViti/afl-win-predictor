"""
This file contains utility functions for combining Parquet files
produced by `data_extraction.R`. 

These functions provide a consistent and reusable approach
for merging datasets such as fixtures, players, or teams.
"""
# libraries
import polars as pl
from collections import defaultdict
from src.config import PROJECT_ROOT

# Functions
def check_parquet_col_dtype(data_path: str) -> None:
    """
    Inspect Parquet files for column data type inconsistencies.

    This function scans all `.parquet` files within the specified
    directory and compares the data types of columns across files.
    It identifies columns that have conflicting dtypes between files
    (e.g., `Int32` in one file and `Float64` in another) and prints
    those mismatches to the console.

    Args:
        data_path (str): Relative path (from PROJECT_ROOT) containing Parquet files
        to inspect, e.g. "data/squiggle_fixture".

    Returns:
        None
    """
    # get a list of all parquet files in specificed data path
    parquet_files = list((PROJECT_ROOT / data_path).glob("*.parquet"))

    # Dictionary to store dtypes per file
    file_column_types = {}

    for f in parquet_files:
        df = pl.read_parquet(f)
        # Record the dtype of each column
        file_column_types[f.name] = {col: df[col].dtype for col in df.columns}

    column_dtype_map = defaultdict(set)
    for types in file_column_types.values():
        for col, dtype in types.items():
            column_dtype_map[col].add(dtype)

    # Print columns that have mismatched types
    print("Columns with dtype mismatches:")
    for col, dtypes in column_dtype_map.items():
        if len(dtypes) > 1:
            print(f"{col}: {dtypes}")

# TODO: consider expanding this function to other dtype issues when pulling other tables
def combine_squiggle_fixtures(folder_path:str, output_path:str) -> None:
    """
    Combine multiple Squiggle fixture Parquet files into a single dataset.

    This function reads all `.parquet` files located in the specified
    `folder_path`, standardizes column ordering and selected data types,
    vertically concatenates the files, and writes the combined dataset
    to `output_path`.

    Args:
        folder_path (str):
            Relative path (from PROJECT_ROOT) containing Squiggle fixture
            Parquet files, e.g. "data/squiggle_fixture".

        output_path (str):
            Relative path (from PROJECT_ROOT) where the combined Parquet
            file will be written, e.g.
            "data/complete_datasets/squiggle_fixture_all_seasons.parquet".

    Returns:
        None
    """
    # get all the parquet files in the data dir
    parquet_files = list((PROJECT_ROOT / folder_path).glob("*.parquet"))

    dfs = []
    for f in parquet_files:
        df = pl.read_parquet(f)
        # Sort columns alphabetically
        df = df.select(sorted(df.columns)) 
        df = df.with_columns(  # fix columns with incorrect dtypes 
            pl.from_epoch("unixtime", time_unit="s"), # convert to correct datetime (s)
            pl.col("timestr").cast(pl.String)
        )
        dfs.append(df)

    # Concatenate all files safely
    df_all = pl.concat(dfs, rechunk=True)

    df_all.write_parquet(PROJECT_ROOT / output_path)

    print(f"squiggle_fixture_all_seasons.parquet saved at {PROJECT_ROOT / output_path}")

# Main Function # TODO: write a main function here 
folder_path = "data/squiggle_fixture"
output_path = "data/complete_datasets/squiggle_fixture_all_seasons.parquet"
combine_squiggle_fixtures(folder_path, output_path)
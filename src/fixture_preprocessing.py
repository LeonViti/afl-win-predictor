import polars as pl
from src.config import PROJECT_ROOT
from collections import defaultdict

# Functions
def check_parquet_col_dtype(data_path: str) -> None:
    """
    Function to check for a dtype mismatch in data files. 

    Args:
        data_path (str): path to parquet files (e.g. data)
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


# Code 


parquet_path = PROJECT_ROOT / "data/*.parquet"
df = pl.read_parquet(parquet_path, allow_missing_columns=True)

df = (
    pl.scan_parquet(PROJECT_ROOT / "data/*.parquet",  missing_columns="insert")
      .with_columns(
          pl.col("unixtime").cast(pl.Int64),
          pl.col("timestr").cast(pl.String)
      )
      .collect()
)
df.write_parquet(PROJECT_ROOT / "data/all_seasons.parquet")


# test for parquet files
# get all the parquet files in the data dir
parquet_files = list((PROJECT_ROOT / "data").glob("*.parquet"))

dfs = []
for f in parquet_files:
    df = pl.read_parquet(f)
    # Ensure unixtime is always Int64
    if "unixtime" in df.columns:
        df = df.with_columns(
            pl.col("unixtime").cast(pl.Float64),
            pl.col("timestr").cast(pl.String)
        )
    dfs.append(df)

# Concatenate all files safely
df_all = pl.concat(dfs, rechunk=True)




fixture_1899 = pl.read_parquet(path)

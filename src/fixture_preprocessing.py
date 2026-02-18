import polars as pl

path = "C:/Users/leon_/Documents/personal_projects/afl-win-predictor/data/*.parquet"
df = pl.read_parquet(path, allow_missing_columns=True)


df = (
    pl.scan_parquet(path)
      .with_columns(
          pl.col("unixtime").cast(pl.Int64)
      )
      .collect()
)
df.write_parquet("data/all_seasons.parquet")


fixture_1899 = pl.read_parquet(path)

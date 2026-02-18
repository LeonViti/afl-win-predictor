# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Imports 
import polars as pl
from src.config import PROJECT_ROOT

# Functions

# Code 
path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
df = pl.read_parquet(path)
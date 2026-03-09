"""
NOTE: Level modelling occurs will be at the match level 
where each row represents two teams, a home and an away team.
"""

# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Imports 
import json
import polars as pl
from src.config import PROJECT_ROOT

# Functions
def create_venue_location_feat(df: pl.DataFrame) -> pl.DataFrame:
    """
    
    """
    # Load JSON as a Python dict
    # TODO: make Path a variable for this function 
    with open(PROJECT_ROOT / "src/reference/venue_location_mapping.json") as f:
        venue_to_location = json.load(f)

    # Convert dict to Polars DataFrame
    venue_map = pl.DataFrame({
        "venue": list(venue_to_location.keys()),
        "venue_location": list(venue_to_location.values())
    })

    # Join to your fixture DataFrame
    venue_loc_df = df.join(venue_map, on="venue", how="left")

    return venue_loc_df

# TODO: make venue_tz_map and team_tz_map paths then add appropriate debugging, repeat for above function 
def create_timezone_feats(
        venue_tz_map: pl.DataFrame, 
        team_tz_map: pl.DataFrame, 
        df: pl.DataFrame
) -> pl.DataFrame:
    """
    Create timezone-related features for home and away teams.

    Args:
        venue_tz_map (pl.DataFrame): Mapping of venue locations to timezone minutes.
        team_tz_map (pl.DataFrame): Mapping of team names to timezone minutes.
        df (pl.DataFrame): Matches dataframe with columns 'hteam', 'ateam', 'venue_location'.

    Returns:
        pl.DataFrame: Original dataframe with additional timezone features:
            - home_tz_diff_min
            - away_tz_diff_min
            - home_tz_shift_min
            - away_tz_shift_min
            - tz_shift_advantage
    """
    # join the timezone features to the dataframe
    tz_df = (
        df
        .join(
            team_tz_map.select(["team", "team_tz_min", "home_state"])
                    .rename({"team": "hteam", "team_tz_min": "hteam_tz_min", "home_state": "hteam_home_state"}),
            on="hteam",
            how="left"
        )
        .join(
            team_tz_map.select(["team", "team_tz_min", "home_state"])
                    .rename({"team": "ateam", "team_tz_min": "ateam_tz_min", "home_state": "ateam_home_state"}),
            on="ateam",
            how="left"
        )
        .join(
            venue_tz_map.select(["venue_location", "venue_location_tz_min"]),
            on="venue_location",
            how="left"
        )
    )

    # collect null values and check if any NULL's are produced
    nulls = tz_df.select([
        pl.col("hteam_tz_min").is_null().sum().alias("null_home_tz"),
        pl.col("ateam_tz_min").is_null().sum().alias("null_away_tz"),
        pl.col("venue_location_tz_min").is_null().sum().alias("null_venue_tz"),
    ]).to_dict(as_series=False)

    if any(v[0] > 0 for v in nulls.values()):
        raise ValueError(f"Null values found in timezone mapping: {nulls}")
    else:
        print("No NULL values produced after timezone feats") # TODO: Change this to a log

    # create the timezone features 
    tz_df = tz_df.with_columns([
        (pl.col("venue_location_tz_min") - pl.col("hteam_tz_min")).alias("home_tz_diff_min"),
        (pl.col("venue_location_tz_min") - pl.col("ateam_tz_min")).alias("away_tz_diff_min"),
    ])

    tz_df = tz_df.with_columns([ # tz shift is the absolute value of the timezone shift
        pl.col("home_tz_diff_min").abs().alias("home_tz_shift_min"),
        pl.col("away_tz_diff_min").abs().alias("away_tz_shift_min"),
    ])
    # let tz_shift_advantage represent the timezone advantage from the home team
    tz_df = tz_df.with_columns((pl.col("away_tz_shift_min") - pl.col("home_tz_shift_min")).alias("tz_shift_advantage"))

    return tz_df

######################################
# Code 
######################################
def compute_fixture_features(path):
    # load in data
    path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
    fixture_df = pl.read_parquet(path)

    # create column venue_location to represent the state or country (if played outside Aus) of the match
    df_clean = create_venue_location_feat(fixture_df)

    # correctly assign datatypes for localtime column
    df_clean = df_clean.with_columns(
        pl.col("localtime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("localtime_dt")
    )

    # CREATE TIMEZONE FEATS
    # load timezone mappings 
    venue_tz_map = pl.read_csv(PROJECT_ROOT / "src/reference/venue_tz_map.csv")
    team_tz_map = pl.read_csv(PROJECT_ROOT / "src/reference/team_tz_map.csv")
    df_clean = create_timezone_feats(venue_tz_map, team_tz_map, df_clean)

    # CREATE is_interstate_game FLAG (home_state != away_state)
    df_clean = df_clean.with_columns([
        (pl.col("hteam_home_state") != pl.col("ateam_home_state"))
            .cast(pl.Int8)
            .alias("is_interstate_game")
    ])

    # create time_of_day column
    df_clean = df_clean.with_columns(
        pl.when(pl.col("localtime_dt").dt.hour().is_between(5, 11))
            .then(pl.lit("morning")) # 5am >= morning < 12pm
        .when(pl.col("localtime_dt").dt.hour().is_between(12, 16))
            .then(pl.lit("afternoon")) # 12pm >= afternoon < 5pm
        .when(pl.col("localtime_dt").dt.hour().is_between(17, 20))
            .then(pl.lit("evening")) # 5pm >= afternoon < 9pm
        .otherwise(pl.lit("night")) # 9pm >= night < 5am
        .alias("time_of_day")
    )

    # create day_of_week column
    df_clean = df_clean.with_columns(
        pl.col("localtime_dt")
            .dt.strftime("%A")
            .str.to_lowercase()
            .alias("day_of_week")
    )

    # create is_weekend flag
    df_clean = df_clean.with_columns(
        pl.col("day_of_week")
        .is_in(["saturday", "sunday"])
        .cast(pl.Int8)
        .alias("is_weekend_game")
    )

    # CREATE WINDOWED FEATS # TODO: make into function 
    # melt home and away sides to long format (team lvl); store if the team won or lost
    home_df = df_clean.select([ # select all home team games
        pl.col("localtime_dt"),
        pl.col("hteam").alias("team"),
        # win = 1 if wins, 0.5 if draw, 0 if loss
        pl.when(pl.col("hscore") > pl.col("ascore")).then(1.0) 
        .when(pl.col("hscore") == pl.col("ascore")).then(0.5)
        .otherwise(0.0)
        .alias("win")
    ])

    away_df = df_clean.select([  # select all away team games
        pl.col("localtime_dt"),
        pl.col("ateam").alias("team"),
        # win = 1 if away wins, 0.5 if draw, 0 if loss
        pl.when(pl.col("ascore") > pl.col("hscore")).then(1.0) 
        .when(pl.col("ascore") == pl.col("hscore")).then(0.5)
        .otherwise(0.0)
        .alias("win")
    ])

    # concatenate teams so each row represents one team and sort by date
    team_df = pl.concat([home_df, away_df]).sort(["team", "localtime_dt"])  # ensure correct order

    team_df = team_df.with_columns([
        # shift wins by 1 to exclude current game for each team
        pl.col("win").shift(1).over("team").alias("win_lag1") 
    ])

    # rolling sum/mean for last 5 games (excluding current)
    team_df = team_df.with_columns([
        pl.col("win_lag1")
        .rolling_mean(window_size=5, min_samples=1)
        .over("team")  # ensures rolling is per team
        .alias("last5_win_rate")
    ])

    # Fill nulls with 0 as events are rare
    team_df = team_df.with_columns([
        pl.col("win_lag1").fill_null(0),
        pl.col("last5_win_rate").fill_null(0)
    ])

    # Last 5 games win rate per team
    win_rate = team_df.select(["localtime_dt", "team", "last5_win_rate"])

    # Merge home features
    df_clean = df_clean.join(
        win_rate.rename({"team":"hteam", "last5_win_rate":"hlast5_win_rate"}),
        on=["localtime_dt", "hteam"],
        how="left"
    )

    # Merge away features
    df_clean = df_clean.join(
        win_rate.rename({"team":"ateam", "last5_win_rate":"alast5_win_rate"}),
        on=["localtime_dt", "ateam"],
        how="left"
    )

    return df_clean

# drop redundant columns NOTE: the tz column is only for adjusting timezones, it is not useful here
# df_clean = df_clean.drop(['timestr', 'unixtime'])

##########################
# EDA 
##########################
# EDA: check for null values
# print("Null Values:")
# df_clean.select(pl.all().null_count())
# # filter rows where column is null
# # df_clean.filter(pl.col("venue_location").is_null())

# # EDA: display the total games played in each location
# pl.Config.set_tbl_rows(20) # set the number of rows to be displayed in the interactive terminal
# # pl.Config.set_tbl_cols(20) # set the number of columns to be displayed in the interactive terminal
# pl.Config.set_fmt_str_lengths(50)
# df_clean["venue_location"].value_counts().sort(by="count", descending=True)
# df_clean["tz"].value_counts().sort(by="count", descending=True)
# df_clean["ateam"].value_counts().sort(by="count", descending=True)



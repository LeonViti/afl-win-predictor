"""
NOTE: Level modelling occurs will be at the match level 
where each row represents two teams, a home and an away team.
"""

# FOR INTERACTIVE SESSION comment out later
# import os
# os.chdir(r"/home/lv/Documents/projects/afl-win-predictor")

# Imports 
import json
import polars as pl
from src.config import PROJECT_ROOT

# Functions
def create_venue_location_feat(df: pl.DataFrame) -> pl.DataFrame:
    """
    Maps AFL venues to their respective geographic locations (e.g., 'MCG' -> 'VIC').

    Args:
        df: Polars DataFrame containing a 'venue' column.

    Returns:
        A Polars DataFrame with an additional 'venue_location' column.
    """
    # Load JSON as a Python dict
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

def calc_short_long_term_feats(
    df: pl.DataFrame, 
    lag_feat: str,
    feat_suffix: str
) -> pl.DataFrame:
    """
    df = team_df
    lag_feat = win_lag1
    feat_suffix = win_rate
    """
    # rolling sum/mean for last 3, 5, 10, 20 games (excluding current)
    df = df.with_columns([
        pl.col(lag_feat)
            .rolling_mean(window_size=3, min_samples=1)
            .over("team")
            .alias(f"last3_{feat_suffix}"),

        pl.col(lag_feat)
            .rolling_mean(window_size=5, min_samples=1)
            .over("team")
            .alias(f"last5_{feat_suffix}"),

        pl.col(lag_feat)
            .rolling_mean(window_size=10, min_samples=1)
            .over("team")
            .alias(f"last10_{feat_suffix}"),

        pl.col(lag_feat)
            .rolling_mean(window_size=20, min_samples=1)
            .over("team")
            .alias(f"last20_{feat_suffix}"),
    ])
    
    return df

def rename_windowed_features(
    df: pl.DataFrame,
    lagged_feats: list[str],
    prefix: str,
    suffixes: list[str]
) -> pl.DataFrame:
    """
    Renames team-level lagged features so they can be joined as home or away features.

    Args:
        df: dataframe containing team features
        lagged_feats: lagged features e.g. ["win_lag1", "score_lag1"]
        prefix: team prefix ("h" or "a")
        suffixes: rolling feature suffixes e.g. ["win_rate", "avg_score"]
    """

    rename_dict = {"team": f"{prefix}team"}

    for lagged_feat in lagged_feats:
        rename_dict[lagged_feat] = f"{prefix}{lagged_feat}"

    for suffix in suffixes:
        for window in [3, 5, 10, 20]:
            rename_dict[f"last{window}_{suffix}"] = f"{prefix}last{window}_{suffix}"

    return df.rename(rename_dict)

def subtract_home_away_features(
    df: pl.DataFrame, 
    general_col_name: str
) -> pl.DataFrame:
    """
    Subtracts the corresponding home and away columns to create a diff feature. 

    Args:
        df: dataframe containing team features
        general_col_name: feature you want to compare, e.g. days_break

    Returns:
        pl.DataFrame with new comparison features. 
    """

    df = df.with_columns([
        (pl.col(f"h{general_col_name}") - pl.col(f"a{general_col_name}"))
            .alias(f"diff_{general_col_name}")
    ])

    return df

def subtract_home_away_windowed_features(
    df: pl.DataFrame, 
    windows: list[int], 
    suffix: str
) -> pl.DataFrame:
    """
    Subtracts the corresponding home and away columns for windowed features. 

    Args:
        df: dataframe containing team home and away features
        windows: list of time windows to iterate over.
        suffix: general name of column you want to target e.g. win_rate.
        general_col_name: feature you want to compare, e.g. days_break

    Returns:
        pl.DataFrame with new comparison features. 
    """

    df = df.with_columns([
        (pl.col(f"hlast{w}_{suffix}") - pl.col(f"alast{w}_{suffix}"))
            .alias(f"diff_last{w}_{suffix}")
        for w in windows
    ])

    return df

######################################
# Code 
######################################
def compute_fixture_features(path):
    # load in data
    # path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
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

    # TODO: create an is home ground flag: hteam_ground == venue

    # CREATE WINDOWED FEATS # TODO: make into function 
    # melt home and away sides to long format (team lvl); store if the team won or lost
    home_df = df_clean.select([ # select all home team games
        pl.col("localtime_dt"),
        pl.col("hteam").alias("team"),
        pl.col("hscore").alias("score"),
        (pl.col("hscore") - pl.col("ascore")).alias("margin"),
        # win = 1 if wins, 0.5 if draw, 0 if loss
        pl.when(pl.col("hscore") > pl.col("ascore")).then(1.0) 
        .when(pl.col("hscore") == pl.col("ascore")).then(0.5)
        .otherwise(0.0)
        .alias("win")
    ])

    away_df = df_clean.select([  # select all away team games
        pl.col("localtime_dt"),
        pl.col("ateam").alias("team"),
        pl.col("ascore").alias("score"),
        (pl.col("hscore") - pl.col("ascore")).alias("margin"),
        # win = 1 if away wins, 0.5 if draw, 0 if loss
        pl.when(pl.col("ascore") > pl.col("hscore")).then(1.0) 
        .when(pl.col("ascore") == pl.col("hscore")).then(0.5)
        .otherwise(0.0)
        .alias("win")
    ])

    # concatenate teams so each row represents one team and sort by date
    team_df = pl.concat([home_df, away_df]).sort(["team", "localtime_dt"])  # ensure correct order

    # shift stats by 1 to exclude current game for each team
    team_df = team_df.with_columns([
        pl.col("score").shift(1).over("team").alias("score_lag1"),
        pl.col("win").shift(1).over("team").alias("win_lag1"),
        pl.col("margin").shift(1).over("team").alias("margin_lag1"),
        pl.col("localtime_dt").shift(1).over("team").alias("localtime_dt_lag1")
    ])
    
    # Calculate the number of days since the team last played
    team_df = team_df.with_columns(
        ((pl.col("localtime_dt") - pl.col("localtime_dt_lag1")).dt.total_days())
        .clip(0, 20) # for end of season, set values > 20 to 20 instead of 200 days
        .fill_null(20) # Assume a standard week for the very first historical records
        .alias("days_break")
    )
    
    # rolling sum/mean for last 3, 5, 10, 20 games (excluding current)
    team_df = calc_short_long_term_feats(team_df, "win_lag1", "win_rate")
    team_df = calc_short_long_term_feats(team_df, "score_lag1", "avg_score")
    team_df = calc_short_long_term_feats(team_df, "margin_lag1", "avg_margin")

    # TODO: review this exclusion of nulls
    # Fill nulls with 0 as events are rare
    team_df = team_df.fill_null(0)

    # Last 5 games win rate per team
    team_df = team_df.select([
        "localtime_dt", "team", "days_break",
        "win_lag1", "last3_win_rate", "last5_win_rate", "last10_win_rate", "last20_win_rate",
        "score_lag1", "last3_avg_score", "last5_avg_score", "last10_avg_score", "last20_avg_score",
        "margin_lag1", "last3_avg_margin", "last5_avg_margin", "last10_avg_margin", "last20_avg_margin"
    ])

    # rename home and away lagged features for joining
    hteam_df = rename_windowed_features(team_df, ["win_lag1", "score_lag1", "margin_lag1"], "h", ["win_rate", "avg_score", "avg_margin"])
    ateam_df = rename_windowed_features(team_df, ["win_lag1", "score_lag1", "margin_lag1"], "a", ["win_rate", "avg_score", "avg_margin"])

    # rename non-lagged features (currently just days_break)
    hteam_df = hteam_df.rename({"days_break": "hdays_break"})
    ateam_df = ateam_df.rename({"days_break": "adays_break"})

    ###########################
    # MERGE BACK TO DF_CLEAN
    ###########################

    # Merge home features
    df_clean = df_clean.join(
        hteam_df,
        on=["localtime_dt", "hteam"],
        how="left"
    )

    # Merge away features
    df_clean = df_clean.join(
        ateam_df,
        on=["localtime_dt", "ateam"],
        how="left"
    )

    # calculate the difference between home and away team for non-windowed features
    df_clean = subtract_home_away_features(df_clean, "days_break")

    # calculate the difference between home and away team for windowed features
    df_clean = subtract_home_away_windowed_features(df_clean, [3, 5, 10, 20], "win_rate")
    df_clean = subtract_home_away_windowed_features(df_clean, [3, 5, 10, 20], "avg_score")
    df_clean = subtract_home_away_windowed_features(df_clean, [3, 5, 10, 20], "avg_margin")

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

# view the number of days_breaks
# break_counts = (
#     team_df["days_break"]
#     .value_counts()
#     .sort("count", descending=True)
# )
# print(break_counts.head(20))


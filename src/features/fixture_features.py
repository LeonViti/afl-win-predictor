# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Imports 
import polars as pl
from src.config import PROJECT_ROOT

# Functions
def create_venue_location_feat(df: pl.DataFrame) -> pl.DataFrame:
    """
    
    """
    venue_loc_df = df.with_columns(
        pl.when(pl.col('venue').is_in(["Adelaide Hills", "Adelaide Oval", "Norwood Oval", "Football Park", "Barossa Park"])).then(pl.lit("SA"))
        .when(pl.col("venue").is_in(["M.C.G.", "Docklands", "Eureka Stadium", "Kardinia Park", "Marvel Stadium",
                                    "GMHBA Stadium", "Mars Stadium", "Corio Oval", "Brunswick St",
                                    "Princes Park", "Victoria Park", "Junction Oval", "East Melbourne",
                                    "Punt Rd", "Waverley Park", "Windy Hill", "Western Oval", "Glenferrie Oval",
                                    "Arden St", "Moorabbin Oval", "Olympic Park", "Yarraville Oval", "Coburg Oval",
                                    "Toorak Park", "Euroa", "Yallourn"])).then(pl.lit("VIC"))
        .when(pl.col("venue").is_in(["Carrara", "Gabba", "Cazaly's Stadium", "Riverway Stadium", "Brisbane Exhibition"])).then(pl.lit("QLD"))
        .when(pl.col("venue").is_in(["S.C.G.", "Sydney Showground", "Stadium Australia", "Blacktown", "Lake Oval", "Albury"])).then(pl.lit("NSW"))
        .when(pl.col("venue").is_in(["Marrara Oval", "Traeger Park"])).then(pl.lit("NT"))
        .when(pl.col("venue").is_in(["Bellerive Oval", "York Park", "University of Tasmania Stadium", "North Hobart"])).then(pl.lit("TAS"))
        .when(pl.col("venue").is_in(["Manuka Oval", "UNSW Canberra Oval", "Bruce Stadium"])).then(pl.lit("ACT"))
        .when(pl.col("venue").is_in(["Perth Stadium", "Optus Stadium", "Subiaco", "W.A.C.A.", "Hands Oval"])).then(pl.lit("WA"))
        .when(pl.col("venue").is_in(["Jiangwan Stadium", "Adelaide Arena at Jiangwan Stadium"])).then(pl.lit("CHN"))
        .when(pl.col("venue").is_in(["Wellington"])).then(pl.lit("NZL"))
        .otherwise(None)  # otherwise null
        .alias('venue_location')
    )

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
            team_tz_map.select(["team", "team_tz_min"])
                    .rename({"team": "hteam", "team_tz_min": "hteam_tz_min"}),
            on="hteam",
            how="left"
        )
        .join(
            team_tz_map.select(["team", "team_tz_min"])
                    .rename({"team": "ateam", "team_tz_min": "ateam_tz_min"}),
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

# load in data
path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
fixture_df = pl.read_parquet(path)

# create column venue_location to represent the state or country (if played outside Aus) of the match
df_clean = create_venue_location_feat(fixture_df)

# correctly assign datatypes for localtime column
df_clean = df_clean.with_columns(
    pl.col("localtime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("localtime_dt")
)

# load timezone mappings and join them to the dataframe
venue_tz_map = pl.read_csv(PROJECT_ROOT / "src/reference/venue_tz_map.csv")
team_tz_map = pl.read_csv(PROJECT_ROOT / "src/reference/team_tz_map.csv")

# create timezone features
# NOTE: the tz column is only for adjusting timezones, it is not useful here
df_clean = create_timezone_feats(venue_tz_map, team_tz_map, df_clean)

# drop redundant columns 
df_clean = df_clean.drop(['timestr', 'unixtime'])

# TODO: create is_interstate_game flag
# is_interstate_game = home_state != away_state

# TODO: create time_of_day column

# TODO: create is_weekend flag

# TODO: create is_night_game flag 

# TODO: create day_of_week column

# TODO: create month column 

##########################
# EDA 
##########################
# EDA: check for null values
print("Null Values:")
df_clean.select(pl.all().null_count())
# filter rows where column is null
# df_clean.filter(pl.col("venue_location").is_null())

# EDA: display the total games played in each location
pl.Config.set_tbl_rows(20) # set the number of rows to be displayed in the interactive terminal
# pl.Config.set_tbl_cols(20) # set the number of columns to be displayed in the interactive terminal
pl.Config.set_fmt_str_lengths(50)
df_clean["venue_location"].value_counts().sort(by="count", descending=True)
df_clean["tz"].value_counts().sort(by="count", descending=True)
df_clean["ateam"].value_counts().sort(by="count", descending=True)



# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Imports 
import polars as pl
from src.config import PROJECT_ROOT

# Functions
# def preprocess_fixture:
#     # create new column for each state.
#     fixture_clean <- fixture %>% mutate(region = case_when(
#         venue %in% c('Adelaide Hills', 'Adelaide Oval', 'Norwood Oval', 'Football Park') ~ "SA",
#         venue %in% c('M.C.G.', 'Docklands', 'Eureka Stadium', 'Kardinia Park', 'Marvel Stadium',
#                      'GMHBA Stadium', 'Mars Stadium') ~ "VIC",
#         venue %in% c('Carrara', 'Gabba', "Cazaly's Stadium", "Riverway Stadium") ~ "QLD",
#         venue %in% c('S.C.G.', 'Sydney Showground', 'Stadium Australia', 'Blacktown') ~ "NSW",
#         venue %in% c('Marrara Oval', 'Traeger Park') ~ 'NT',
#         venue %in% c('Bellerive Oval', 'York Park', 'University of Tasmania Stadium') ~ "TAS",
#         venue %in% c('Manuka Oval', 'UNSW Canberra Oval') ~ 'ACT',
#         venue %in% c('Perth Stadium', 'Optus Stadium', 'Subiaco') ~ 'WA',
#         venue %in% c('Jiangwan Stadium', 'Adelaide Arena at Jiangwan Stadium') ~ 'CHN',
#         venue %in% c('Wellington') ~ 'NZL',
#         TRUE ~ NA_character_  # set NA for all other observations
#     ))

    
#     fixture_clean$date <- as.Date(fixture_clean$localtime)
#     fixture_clean$time <- format(ymd_hms(fixture_clean$localtime), "%H:%M:%S")
#     fixture_clean$home_win <- ifelse(fixture_clean$hscore > fixture_clean$ascore, 1, 0) 
#     fixture_clean$away_win <- ifelse(fixture_clean$hscore < fixture_clean$ascore, 1, 0)
#     fixture_clean$hdiff <- fixture_clean$hscore - fixture_clean$ascore
    
#     # select specific rows
#     fixture_clean <- select(fixture_clean, year, round, date, time, region, venue, hteam, ateam, hscore, ascore,
#     hdiff, is_grand_final, is_final, home_win, away_win, id)
    
#     return(fixture_clean)

# Code 
path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
fixture_df = pl.read_parquet(path)

# create column venue_location to represent the state or country (if played outside Aus) of the match
df_clean = fixture_df.with_columns(
    pl.when(pl.col('venue').is_in(["Adelaide Hills", "Adelaide Oval", "Norwood Oval", "Football Park"])).then(pl.lit("SA"))
    .when(pl.col("venue").is_in(["M.C.G.", "Docklands", "Eureka Stadium", "Kardinia Park", "Marvel Stadium",
                                 "GMHBA Stadium", "Mars Stadium", "Corio Oval", "Brunswick St",
                                 "Princes Park", "Victoria Park", "Junction Oval"])).then(pl.lit("VIC"))
    .when(pl.col("venue").is_in(["Carrara", "Gabba", "Cazaly's Stadium", "Riverway Stadium"])).then(pl.lit("QLD"))
    .when(pl.col("venue").is_in(["S.C.G.", "Sydney Showground", "Stadium Australia", "Blacktown", "Lake Oval"])).then(pl.lit("NSW"))
    .when(pl.col("venue").is_in(["Marrara Oval", "Traeger Park"])).then(pl.lit("NT"))
    .when(pl.col("venue").is_in(["Bellerive Oval", "York Park", "University of Tasmania Stadium"])).then(pl.lit("TAS"))
    .when(pl.col("venue").is_in(["Manuka Oval", "UNSW Canberra Oval"])).then(pl.lit("ACT"))
    .when(pl.col("venue").is_in(["Perth Stadium", "Optus Stadium", "Subiaco"])).then(pl.lit("WA"))
    .when(pl.col("venue").is_in(["Jiangwan Stadium", "Adelaide Arena at Jiangwan Stadium"])).then(pl.lit("CHN"))
    .when(pl.col("venue").is_in(["Wellington"])).then(pl.lit("NZL"))
    .otherwise(None)  # otherwise null
    .alias('venue_location')
)

# correctly assign datatypes for colums

# TODO: correctly assign the correct datatype for the localtime column

# TODO: create a feature from the Timezone (Tz) column


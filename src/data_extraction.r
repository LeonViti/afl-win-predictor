library(fitzRoy) # for loading of data
library(arrow) # for storing data as a parquet
# get the individual player data
player_stats = fetch_player_stats_afl(season = 2023)

# get the squiggle fixture data
fixture_23 <- fetch_fixture_squiggle(2023)
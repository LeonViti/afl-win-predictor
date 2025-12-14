# libraries 
library(fitzRoy) # for loading of data
library(arrow) # for storing data as a parquet

# Loading of Data
# get the individual player data
player_stats_2023 = fetch_player_stats_afl(season = 2023)

# get the squiggle fixture data
fixture_23 <- fetch_fixture_squiggle(2023)

# save as Parquet
write_
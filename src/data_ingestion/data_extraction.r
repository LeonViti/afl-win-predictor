# libraries 
library(fitzRoy) # for loading of data
library(arrow) # for storing data as a parquet

# TODO: turn this into a function
# Loading of Data
# get the squiggle fixture data
# NOTE: 1899 is the first date that contains data

# iterate over all seasons and save as a parquet
for (i in 1899:2025) {
    # Fetch fixture data for year i
    fixture <- fetch_fixture_squiggle(i)

    # Save to Parquet
    write_parquet(fixture, paste0("data/squiggle_fixture/squiggle_fixture_", i, ".parquet"))
}

get_squiggle_fixture <- function(start_season, end_season) {
    """
    Pulls the squiggle fixture data between date range (inclusive).

    Args:
        start_season (int): initial season to follect data from (e.g. 1899)
        end_season (str): final season to collect data from (e.g. 2025)
    """
    for (i in start_season:end_season) {
        # Fetch fixture data for year i
        fixture <- fetch_fixture_squiggle(i)

        # Save to Parquet
        write_parquet(fixture, paste0("data/squiggle_fixture/squiggle_fixture_", i, ".parquet"))
    }
}

get_squiggle_fixture(1899, 2025)

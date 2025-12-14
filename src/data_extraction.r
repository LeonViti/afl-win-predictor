# libraries 
library(fitzRoy) # for loading of data
library(arrow) # for storing data as a parquet

# Loading of Data
# get the squiggle fixture data
# NOTE: 1899 is the first date that contains data

# iterate over all seasons and save as a parquet
for (i in 1899:2025) {
    # Fetch fixture data for year i
    fixture <- fetch_fixture_squiggle(i)

    # Save to Parquet
    write_parquet(fixture, paste0("data/squiggle_fixture_", i, ".parquet"))
}

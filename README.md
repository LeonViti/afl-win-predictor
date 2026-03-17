# afl-win-predictor
This repo applies machine learning to predict whether a team will win or lose a match in Australian Rules Football (AFL). Includes data extraction, preprocessing, feature engineering, model training, evaluation, and win/loss classification.  

# Data Pipeline  
data_ingestion -> features -> modelling -> evalutation  

# Setup
## Dependencies
This project uses [Task](https://taskfile.dev/) for environment setup and [UV](https://docs.astral.sh/uv/) for virtual environement management. Ensure they are installed before running this project. 

## Data Folder  
To create data folders to save squiggle data to run the following.

```bash
task setup-data-folders
``` 

## R Environment 

Run `sudo apt install -y libcurl4-openssl-dev`

Run `Rscript -e 'renv::install(c("jsonlite", "rlang"))'` from terminal to install packages 

Run `Rscript -e 'renv::restore()'` from terminal to install the required R packages.

**Note**: This can take a while (up to 20min) and will look stuck on `Installing arrow ...`. It is not stuck, just give it time. 

All data is obtained using the fitzRoy data library in R. 

## Python Environment
Ensure UV is installed.
Run `uv sync` to setup the python virtual env. 
Run `.venv\Scripts\activate` to activate UV environment.  

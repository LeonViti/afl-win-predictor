# afl-win-predictor
This repo applies machine learning to predict whether a team will win or lose a match in Australian Rules Football (AFL). Includes data extraction, preprocessing, feature engineering, model training, evaluation, and win/loss classification.  

# Data Pipeline  
data_ingestion -> features -> modelling -> evalutation  

# Setup
## Data Folder  
Go to the Root dir, then run Run 

```bash
mkdir -p data/squiggle_fixture
```
```bash
mkdir data/complete_datasets
```

## R Environment 
Run `renv::restore()` in the R interactive terminal CTRL+ENTER to setup R environment. All data is obtained using the fitzRoy data library. 

## Python Environment
Ensure UV is installed.
Run `uv sync` to setup the python virtual env. 
Run `.venv\Scripts\activate` to activate UV environment.  

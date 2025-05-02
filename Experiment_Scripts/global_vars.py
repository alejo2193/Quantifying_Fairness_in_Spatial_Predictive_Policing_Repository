# global_vars.py
#
# This file defines global parameters and configurations used across various notebooks
# and scripts in the project, particularly for simulations and data processing.

import datetime
import os

#### Global Parameters ####

# --- Region of Study Parameters ---
# These define the spatial boundaries and grid resolution for the analysis area.
x_min = 0       # Minimum x-coordinate of the study region (e.g., in projected units like meters)
x_max = 1       # Maximum x-coordinate of the study region
y_min = 0       # Minimum y-coordinate of the study region
y_max = 1       # Maximum y-coordinate of the study region
grid_size = 0.20 # Size of the grid cells (e.g., in projected units like meters or as a fraction of the region size)

# --- Time Parameters ---
# These define the temporal range and units for the data and simulations.
f_inicial = datetime.datetime(2028, 1, 1, 0, 0) # Start date and time for the overall data/simulation period
f_final = datetime.datetime(2031, 1, 31, 0, 0) # End date and time for the overall data/simulation period
days_time_unit = 1 # Definition of a time unit in days (e.g., 1 day)
f_final_train = datetime.datetime(2030, 1, 1, 0, 0) # End date and time for the training data split
f_final_test = datetime.datetime(2031, 1, 1, 0, 0) # End date and time for the testing data split (start of test period)
f_final_val = datetime.datetime(2031, 1, 31, 0, 0) # End date and time for the validation data split (end of validation period)

n_events_day = 10 ## Expected average number of events per day (for simulations)

# --- Path Configurations ---
# These define the directory paths for storing different types of data and model outputs.
# NOTE: These paths are relative to the project's base directory.
# Ensure these directories exist and are writable before running scripts that use them.

## Simulated data directory
dir_sims = "DATOS/SIMULACIONES/"

## Split data directory (for train/test/validation sets)
dir_split = "DATOS/TRAIN_TEST/"

### Trained models directory
dir_models = "DATOS/MODELOS/"

# Example usage (commented out):
# To access a file: os.path.join(dir_sims, "simulated_events.pkl")
# To create directories if they don't exist:
# os.makedirs(dir_sims, exist_ok=True)
# os.makedirs(dir_split, exist_ok=True)
# os.makedirs(dir_models, exist_ok=True)

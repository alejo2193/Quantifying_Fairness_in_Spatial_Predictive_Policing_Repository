# models/model_initialization.py
#
# This file contains utility functions for initializing different crime prediction
# models from the open_cp library with their respective parameters.

import numpy as np # Imported, but not directly used in these initialization functions.
import open_cp # Core open_cp library
import open_cp.sources.sepp # Imports SEPP data source utilities (not directly used here but might be needed elsewhere)
import datetime # For handling date and time objects.
from math import sqrt # Imported, but not directly used in these initialization functions.
import numpy as np # Redundant import.

# Import specific model implementations from open_cp.
import open_cp.sepp as sepp # SEPP model implementation
import open_cp.naive as naive # Naive model implementation
import open_cp.kde as kde # KDE model implementation

# Import pickle for loading/saving objects (imported, but not used in these functions).
import pickle as pkl
# Import matplotlib.dates for date formatting (imported, but not used in these functions).
import matplotlib.dates
# Import various open_cp modules (some might be redundant imports depending on other files).
import open_cp.plot
import open_cp.geometry
import open_cp.predictors
import open_cp.sources.sepp # Redundant import.
from open_cp import evaluation # Evaluation module (imported, but not used in these functions).

# Import clear_output for clearing output in interactive environments (imported, but not used in these functions).
from IPython.display import clear_output


# region = open_cp.RectangularRegion(xmin=0, xmax=200, ymin=0, ymax=200) # Commented out: Example region definition


# Define a function to initialize the Naive model (CountingGridKernel).
# Takes the training data (TimedPoints object).
# Returns an initialized Naive model predictor object.
# def NAIVE_MODEL (data,grid_size,year_p,month_p,day_p): # Commented out original signature
def NAIVE_MODEL (data):
    # Assign the input data to a variable (redundant assignment).
    timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))] # Commented out filtering

    # Initialize the Naive model predictor (ScipyKDE).
    # NOTE: The function name is NAIVE_MODEL, but it initializes naive.ScipyKDE.
    # This might be confusing; consider renaming the function or using a different Naive model class if intended.
    predictor = naive.ScipyKDE()
    # Set the training data for the predictor.
    predictor.data = timed_points
    # Commented out lines for prediction and grid conversion:
    # prediction = predictor.predict()
    # grid = open_cp.data.Grid(xsize=grid_size, ysize=grid_size,xoffset=min(timed_points.xcoords), yoffset=min(timed_points.ycoords))
    # gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(prediction, region, grid_size)

    # Return the initialized Naive model predictor object.
    return predictor


# Define a function to initialize the KDE model.
# Takes training data, region, grid size, and a time kernel.
# Returns an initialized KDE model predictor object.
# def KDE_MODEL (data,grid_size,kernel_time,sampless,year_p,month_p,day_p): # Commented out original signature
def KDE_MODEL (data,region,grid_size,kernel_time):
    # Assign the input data to a variable (redundant assignment).
    timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))] # Commented out filtering

    # Initialize the KDE model with region and grid size.
    predictor = kde.KDE(region=region, grid_size=grid_size)
    # Set the time kernel for the KDE model.
    predictor.time_kernel = kernel_time
    # Set the space kernel for the KDE model (using a Gaussian base provider).
    predictor.space_kernel = kde.GaussianBaseProvider()
    # Set the training data for the predictor.
    predictor.data = timed_points
    # Commented out line for prediction:
    # gridpred = predictor.predict(samples=sampless)

    # Return the initialized KDE model predictor object.
    return predictor


# Define a function to initialize the SEPP model trainer.
# Takes training data, number of training iterations, time cutoff in hours, and a space cutoff.
# Returns a trained SEPP model predictor object.
# def SEPP_MODEL (data,iteration,grid_size,hourss,cutoff,year_p,month_p,day_p): # Commented out original signature
def SEPP_MODEL (data,iteration,hourss,cutoff):
    # Assign the input data to a variable (redundant assignment).
    # timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))] # Commented out filtering

    # Initialize the SEPPTrainer.
    trainer = sepp.SEPPTrainer()
    # Set the training data for the trainer.
    trainer.data = data # Use the input 'data' directly
    # Set the space cutoff parameter for the trainer.
    trainer.space_cutoff = cutoff
    # Set the time cutoff parameter for the trainer using a timedelta based on hours.
    trainer.time_cutoff = datetime.timedelta(hours=hourss)

    # Train the SEPP model using the specified number of iterations.
    predictor = trainer.train(iterations=iteration)

    # Commented out original training loop with error handling:
    # Entrenamiento
    # iteration=40
    # while (iteration>=2):
    #   try:
    #     print('Empieza P, iteración: ',iteration)
    #     predictor = trainer.train(iterations=iteration)
    #     print('Convergencia P, iteración: ',iteration)
    #     break
    #   except:
    #     print('No convergencia P, iteración: ',iteration)
    #     iteration-=2
    #     clear_output(wait=True)
    # print('Convergencia P, iteración: ',iteration)

    # Set the training data on the resulting predictor object.
    # This might be necessary for the predict method depending on the open_cp version.
    predictor.data = data # Use the input 'data' directly

    # Commented out lines for prediction and grid conversion:
    # dates = datetime.datetime(year_p,month_p,day_p)
    # predictions = predictor.predict(dates)
    # gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(predictions, region, grid_size, grid_size)

    # Return the trained SEPP model predictor object.
    return predictor

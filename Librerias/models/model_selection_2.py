# models/model_init_plotting.py (or similar utility file)
#
# This file contains functions to initialize different crime prediction models
# and generate plots of their predictions alongside real event data for a specific date.

import matplotlib.pyplot as plt # For plotting
import numpy as np # For numerical operations
import open_cp # Core open_cp library
import open_cp.sources.sepp # Imports SEPP data source utilities
import datetime # For handling date and time objects.
from math import sqrt # Imported, but not directly used in these functions.
import numpy as np # Redundant import.
from scipy.stats import wasserstein_distance # Imported, but not directly used in these functions.
#import time # Commented out import, possibly for timing operations


# Import specific model implementations from open_cp.
import open_cp.naive as naive # Naive model implementation
import open_cp.kde as kde # KDE model implementation


import pickle as pkl # For loading/saving objects (imported, but not used in these functions).
import matplotlib.dates # For date formatting (imported, but not used in these functions).


import open_cp # Redundant import.
import open_cp.plot # Plotting utilities (imported, but not directly used for the main plotting logic here).
import open_cp.geometry # Geometry utilities (imported, but not directly used here).
import open_cp.predictors # Predictor utilities (used for GridPredictionArray).
import open_cp.sources.sepp # Redundant import.
#import sepp.sepp_grid # Commented out import


import open_cp.seppexp as seppexp # SEPP Exponential model implementation (imported, but not directly used here).
from open_cp import evaluation # Evaluation module (imported, but not used in these functions).
import open_cp.sepp_2 as sepp # Another SEPP implementation (used for SEPPTrainer).

# Import specific SEPP modules (likely custom implementations).
import sepp.sepp_grid_space # SEPP grid space implementation (used for SEPP_GRID_SPACE function).
import sepp.sepp_full # Imported, but not directly used here.
import sepp.sepp_fixed # Imported, but not directly used here.
import sepp.sepp_grid # Imported, but not directly used here.


# Define a fixed rectangular region for plotting.
# NOTE: This region is hardcoded and might need to be dynamic based on the data or analysis area.
region = open_cp.RectangularRegion(xmin=0, xmax=200, ymin=0, ymax=200)


# Define a function to initialize the Naive model, predict, and plot.
# Takes training data, grid size, and prediction date components.
# NOTE: This function performs initialization, prediction, and plotting, which might be less modular.
def NAIVE_MODEL (data,grid_size,year_p,month_p,day_p):
    # Filter training data to include events before the prediction date.
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    # Initialize the Naive model predictor (ScipyKDE).
    predictor = naive.ScipyKDE()
    # Set the training data for the predictor.
    predictor.data = timed_points
    # Generate the continuous prediction.
    prediction = predictor.predict()

    # Convert the continuous prediction to a GridPredictionArray.
    # Defines a grid based on grid_size and offsets from the training data's min coords.
    # NOTE: Using min(timed_points.xcoords) and min(timed_points.ycoords) for offset might not align with a global grid.
    # NOTE: The 'region' variable used here is the hardcoded global one, not passed as an argument.
    grid = open_cp.data.Grid(xsize=grid_size, ysize=grid_size,xoffset=min(timed_points.xcoords), yoffset=min(timed_points.ycoords))
    gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(prediction, region, grid_size) # NOTE: grid_size appears twice


    # Filter the original data to get real events specifically on the prediction date.
    data_Prediccion = data[data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0)]

    # Plot the prediction grid and real events.
    fig, ax = plt.subplots(figsize=(10,10))
    # Set plot limits based on the hardcoded global region.
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    # Plot the intensity matrix from the grid prediction using pcolormesh.
    m = ax.pcolormesh(*gridpred.mesh_data(), gridpred.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax) # Commented out: Original colorbar line
    # Scatter plot the real events on the same axes.
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    # Set the plot title and axis labels.
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    # Add a color bar.
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    # NOTE: plt.show() is missing here to display the plot.


# Define a function to initialize the KDE model, predict, and plot.
# Takes training data, grid size, time kernel parameters, samples, and prediction date components.
# NOTE: This function also performs initialization, prediction, and plotting.
def KDE_MODEL (data,grid_size,kernel_time,sampless,year_p,month_p,day_p):
    # Filter training data to include events before the prediction date.
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    # Initialize the KDE model with the hardcoded global region and grid size.
    predictor = kde.KDE(region=region, grid_size=grid_size)
    # Set the time kernel for the KDE model.
    predictor.time_kernel = kde.ExponentialTimeKernel(kernel_time) # Uses ExponentialTimeKernel with provided parameter.
    # Set the space kernel for the KDE model (using a Gaussian base provider).
    predictor.space_kernel = kde.GaussianBaseProvider()
    # Set the training data for the predictor.
    predictor.data = timed_points
    # Generate the grid prediction using the specified number of samples.
    gridpred = predictor.predict(samples=sampless)

    # Filter the original data to get real events specifically on the prediction date.
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]

    # Plot the prediction grid and real events.
    fig, ax = plt.subplots(figsize=(10,10))

    # Plot the intensity matrix from the grid prediction using pcolor.
    m = ax.pcolor(*gridpred.mesh_data(), gridpred.intensity_matrix)
    # Scatter plot the real events on the same axes.
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, marker="+", color="black")
    # Set the plot title and axis labels.
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    # Add a color bar.
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    # NOTE: plt.show() is missing here to display the plot.


# Define a function to initialize the SEPP model, train, predict, and plot.
# Takes training data, grid size, time cutoff hours, space cutoff, and prediction date components.
# NOTE: This function also performs initialization, training, prediction, and plotting.
def SEPP_MODEL (data,grid_size,hourss,cutoff,year_p,month_p,day_p):
    # Filter training data to include events before the prediction date.
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    # Initialize the SEPPTrainer.
    trainer = sepp.SEPPTrainer() # NOTE: Uses sepp.SEPPTrainer, not seppexp.SEPPTrainer.
    # Set the training data for the trainer.
    trainer.data = timed_points
    # Set space and time cutoff parameters.
    trainer.space_cutoff = cutoff
    trainer.time_cutoff = datetime.timedelta(hours=hourss)

    # Train the SEPP model.
    # NOTE: Number of iterations is not specified here, it will use the default.
    predictor = trainer.train()

    # Set the training data on the resulting predictor object (redundant if already set on trainer).
    predictor.data = timed_points
    # Define the prediction date.
    dates = datetime.datetime(year_p,month_p,day_p)
    # Generate the continuous prediction for the prediction date.
    predictions = predictor.predict(dates)
    # Convert the continuous prediction to a GridPredictionArray.
    # Uses the hardcoded global region and grid size.
    grided = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(predictions, region, grid_size, grid_size) # NOTE: grid_size appears twice
    #grided_bground = open_cp.predictors.grid_prediction_from_kernel(predictor.background_kernel.space_kernel, region, grid_size) # Commented out line

    # Filter the original data to get real events specifically on the prediction date.
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]
    # Plot the prediction grid and real events.
    fig, ax = plt.subplots(figsize=(10,10))
    # Set plot limits based on the hardcoded global region.
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    # Plot the intensity matrix from the grid prediction using pcolormesh.
    m = ax.pcolormesh(*grided.mesh_data(), grided.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax) # Commented out: Original colorbar line
    # Scatter plot the real events on the same axes.
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    # Set the plot title and axis labels.
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    # Add a color bar.
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    # NOTE: plt.show() is missing here to display the plot.


# Define a function to initialize the SEPP_GRID_SPACE model, train, predict, and plot.
# Takes training data, grid size, time cutoff hours, space cutoff, and prediction date components.
# NOTE: This function also performs initialization, training, prediction, and plotting.
def SEPP_GRID_SPACE (data,grid_size,hourss,cutoff,year_p,month_p,day_p):
    # Filter training data to include events before the prediction date.
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    # Initialize time and space kernel providers for SEPP_GRID_SPACE.
    # NOTE: Fixed bandwidths (1 and 20) are hardcoded.
    tk = sepp.sepp_grid_space.FixedBandwidthTimeKernelProvider(1)
    sk = sepp.sepp_grid_space.FixedBandwidthSpaceKernelProvider(20)
    # Initialize the Trainer4 for SEPP_GRID_SPACE with the hardcoded global region and kernels.
    trainer = sepp.sepp_grid_space.Trainer4(region, tk, sk)
    # Set the training data for the trainer.
    trainer.data = timed_points
    # Train the model for the specified prediction date and number of iterations.
    model = trainer.train(datetime.datetime(year_p,month_p,day_p,0,0), iterations=50)

    # Convert the trained model to a predictor object.
    predictor = trainer.to_predictor(model)

    # Set the data on the predictor (redundant if already set on trainer).
    predictor.data = data # NOTE: Uses the full 'data', not just 'timed_points'.

    # Perform prediction for a 1-day window starting from the prediction date.
    # Uses time_samples and space_samples parameters.
    # start = time.clock() # Commented out: Start timing
    grided = predictor.predict(datetime.datetime(year_p,month_p,day_p,0,0), datetime.datetime(year_p,month_p,day_p+1,0,0), time_samples=5, space_samples=-5)
    # print(time.clock() - start) # Commented out: Print elapsed time
    # back = predictor.background_predict(datetime.datetime(year_p,month_p,day_p,0,0), space_samples=-5) # Commented out line

    # Filter the original data to get real events specifically on the prediction date.
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]
    # Plot the prediction grid and real events.
    fig, ax = plt.subplots(figsize=(10,10))
    # Set plot limits based on the hardcoded global region.
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    # Plot the intensity matrix from the grid prediction using pcolormesh.
    m = ax.pcolormesh(*grided.mesh_data(), grided.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax) # Commented out: Original colorbar line
    # Scatter plot the real events on the same axes.
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    # Set the plot title and axis labels.
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    # Add a color bar.
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    # NOTE: plt.show() is missing here to display the plot.

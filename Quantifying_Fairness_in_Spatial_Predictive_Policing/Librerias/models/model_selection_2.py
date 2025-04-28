import matplotlib.pyplot as plt
import numpy as np
import open_cp
import open_cp.sources.sepp
import datetime
from math import sqrt
import numpy as np
from scipy.stats import wasserstein_distance
#import time


import open_cp.naive as naive
import open_cp.kde as kde


import pickle as pkl
import matplotlib.dates


import open_cp
import open_cp.plot
import open_cp.geometry
import open_cp.predictors
import open_cp.sources.sepp
#import sepp.sepp_grid


import open_cp.seppexp as seppexp
from open_cp import evaluation
import open_cp.sepp_2 as sepp

import sepp.sepp_grid_space
import sepp.sepp_full
import sepp.sepp_fixed
import sepp.sepp_grid


region = open_cp.RectangularRegion(xmin=0, xmax=200, ymin=0, ymax=200)


def NAIVE_MODEL (data,grid_size,year_p,month_p,day_p):
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    predictor = naive.ScipyKDE()
    predictor.data = timed_points
    prediction = predictor.predict()


    #Ventana espacial 250x250 metros unicamente por testeo
    grid = open_cp.data.Grid(xsize=grid_size, ysize=grid_size,xoffset=min(timed_points.xcoords), yoffset=min(timed_points.ycoords))
    gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(prediction, region, grid_size)


    #Data original
    data_Prediccion = data[data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0)]

    #Ploteo data
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    m = ax.pcolormesh(*gridpred.mesh_data(), gridpred.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax)
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    

def KDE_MODEL (data,grid_size,kernel_time,sampless,year_p,month_p,day_p):
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]
    
    predictor = kde.KDE(region=region, grid_size=grid_size)
    predictor.time_kernel = kde.ExponentialTimeKernel(kernel_time)
    predictor.space_kernel = kde.GaussianBaseProvider()
    predictor.data = timed_points
    gridpred = predictor.predict(samples=sampless)
    
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]

    fig, ax = plt.subplots(figsize=(10,10))

    m = ax.pcolor(*gridpred.mesh_data(), gridpred.intensity_matrix)
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, marker="+", color="black")
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    
    
    
def SEPP_MODEL (data,grid_size,hourss,cutoff,year_p,month_p,day_p):
    #Inicializando el predictor, los datos de entrenamiento y los cutoff
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]
    
    trainer = sepp.SEPPTrainer()
    trainer.data = timed_points
    trainer.space_cutoff = cutoff
    trainer.time_cutoff = datetime.timedelta(hours=hourss)
    
    #Entrenamiento
    predictor = trainer.train()
    
    #Predicción
    predictor.data = timed_points
    dates = datetime.datetime(year_p,month_p,day_p)
    predictions = predictor.predict(dates) 
    grided = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(predictions, region, grid_size, grid_size)
    #grided_bground = open_cp.predictors.grid_prediction_from_kernel(predictor.background_kernel.space_kernel, region, grid_size)
    
    #Data original
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]
    #Ploteo data
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    m = ax.pcolormesh(*grided.mesh_data(), grided.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax)
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
    
    
def SEPP_GRID_SPACE (data,grid_size,hourss,cutoff,year_p,month_p,day_p):
    #Inicializando el predictor, los datos de entrenamiento y los cutoff
    timed_points=data[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]
    
    tk = sepp.sepp_grid_space.FixedBandwidthTimeKernelProvider(1)
    sk = sepp.sepp_grid_space.FixedBandwidthSpaceKernelProvider(20)
    trainer = sepp.sepp_grid_space.Trainer4(region, tk, sk)
    trainer.data = timed_points
    model = trainer.train(datetime.datetime(year_p,month_p,day_p,0,0), iterations=50)
    
    #Entrenamiento
    predictor = trainer.to_predictor(model)
    
    #Predicción
    
    predictor.data = data
    #start = time.clock()
    grided = predictor.predict(datetime.datetime(year_p,month_p,day_p,0,0), datetime.datetime(year_p,month_p,day_p+1,0,0), time_samples=5, space_samples=-5)
    #print(time.clock() - start)
    #back = predictor.background_predict(datetime.datetime(year_p,month_p,day_p,0,0), space_samples=-5)
    
    #Data original
    data_Prediccion = data[(data.times_datetime()==datetime.datetime(year_p,month_p,day_p,0,0))]
    #Ploteo data
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
    m = ax.pcolormesh(*grided.mesh_data(), grided.intensity_matrix, cmap="jet")
    #fig.colorbar(m, ax=ax)
    ax.scatter(data_Prediccion.xcoords, data_Prediccion.ycoords, alpha=0.5, color='black')
    ax.set_title("Predicción del riesgo a {}".format(str(year_p)+"-"+str(month_p)+"-"+str(day_p)))
    ax.set_xlabel('Cordenada X')
    ax.set_ylabel('Cordenada Y')
    cb = plt.colorbar(m, ax=ax)
    cb.set_label("Relative risk")
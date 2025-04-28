import numpy as np
import open_cp
import open_cp.sources.sepp
import datetime
from math import sqrt
import numpy as np

import open_cp.sepp as sepp
import open_cp.naive as naive
import open_cp.kde as kde

import pickle as pkl
import matplotlib.dates
import open_cp
import open_cp.plot
import open_cp.geometry
import open_cp.predictors
import open_cp.sources.sepp
from open_cp import evaluation

from IPython.display import clear_output


# region = open_cp.RectangularRegion(xmin=0, xmax=200, ymin=0, ymax=200)


# def NAIVE_MODEL (data,grid_size,year_p,month_p,day_p):
def NAIVE_MODEL (data):    
    timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]

    predictor = naive.ScipyKDE()
    predictor.data = timed_points
    #prediction = predictor.predict()
    #grid = open_cp.data.Grid(xsize=grid_size, ysize=grid_size,xoffset=min(timed_points.xcoords), yoffset=min(timed_points.ycoords))
    #gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(prediction, region, grid_size)
    
    return predictor


# def KDE_MODEL (data,grid_size,kernel_time,sampless,year_p,month_p,day_p):
def KDE_MODEL (data,region,grid_size,kernel_time):
    timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]
    
    predictor = kde.KDE(region=region, grid_size=grid_size)
    predictor.time_kernel = kernel_time
    predictor.space_kernel = kde.GaussianBaseProvider()
    predictor.data = timed_points
    #gridpred = predictor.predict(samples=sampless)
    
    return predictor
    
    
# def SEPP_MODEL (data,iteration,grid_size,hourss,cutoff,year_p,month_p,day_p):
def SEPP_MODEL (data,iteration,hourss,cutoff):
    #Inicializando el predictor, los datos de entrenamiento y los cutoff
    timed_points=data#[(data.times_datetime()<datetime.datetime(year_p,month_p,day_p,0,0))]
    
    trainer = sepp.SEPPTrainer()
    trainer.data = timed_points
    trainer.space_cutoff = cutoff
    trainer.time_cutoff = datetime.timedelta(hours=hourss)

    predictor = trainer.train(iterations=iteration)
    
    #Entrenamiento
    #iteration=40
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
    #Predicción
    predictor.data = timed_points
    #dates = datetime.datetime(year_p,month_p,day_p)
    #predictions = predictor.predict(dates) 
    #gridpred = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(predictions, region, grid_size, grid_size)

    return predictor
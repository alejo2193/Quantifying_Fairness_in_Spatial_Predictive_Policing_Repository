import datetime
import os 

#### parametros globales

### region de estudio
x_min=0
x_max=1
y_min=0
y_max=1
grid_size=0.20



# tiempo 
f_inicial=datetime.datetime(2028,1,1,0,0)
f_final=datetime.datetime(2031,1,31,0,0)
days_time_unit=1
f_final_train=datetime.datetime(2030,1,1,0,0)
f_final_test=datetime.datetime(2031,1,1,0,0)
f_final_val=datetime.datetime(2031,1,31,0,0)

n_events_day = 10 ## expected events per day

# paths


## datos simulados
dir_sims="DATOS/SIMULACIONES/"

## split datos
dir_split="DATOS/TRAIN_TEST/"

### paths_models trained
dir_models="DATOS/MODELOS/"


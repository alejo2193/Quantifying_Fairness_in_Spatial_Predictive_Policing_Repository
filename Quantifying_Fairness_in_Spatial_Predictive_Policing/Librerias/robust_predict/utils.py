import numpy as np
import matplotlib.pyplot as plt
import sparse as sp

import multiprocessing
from multiprocessing import Pool
coreCount = multiprocessing.cpu_count()
#pool = max(Pool(coreCount - 4) ,1)


##### np.array
def plot_intensity(intensity,vmax=None):
    fig, ax = plt.subplots(figsize=(11,5))
    im=ax.imshow(intensity, 'jet', interpolation='none', alpha=0.7,vmin=0,vmax=vmax)
    ax.invert_yaxis()
    plt.colorbar(im)

    plt.show(ax)

### coord to index pos array
def get_position(coords,size_m):
    pos=np.floor(coords[::-1].T*np.array(size_m)).astype(int)
    pos[:,0][pos[:,0]==size_m[0]]=size_m[0]-1
    pos[:,1][pos[:,1]==size_m[1]]=size_m[1]-1
    return pos.T



### define neigborhood cells around
def neighborhood_get(posicion,mask):
    ## posicion (x,y)
    filas, columnas = mask.shape
    fila, columna = posicion

    celdas_adyacentes = []

    for i in range(fila-1, fila+2):
        for j in range(columna-1, columna+2):
            if 0 <= i < filas and 0 <= j < columnas and mask[i, j]:
                celdas_adyacentes.append((i, j))

    return celdas_adyacentes



### pos matrix to flatten
def get_flatten_pos(pos_matrix,size_m=(5,5)):
    return np.ravel_multi_index(pos_matrix, size_m)



#### create W as identifier cell
def create_W(filas,columnas,data_shape,data_size):
    ### W must have the size like a vector for each point in data

    m=filas*columnas ## number of covariates 

    #### covariates // for identify cell position
    new_coors=np.indices(data_shape).reshape((3,data_size))

    aux_func = lambda x : get_flatten_pos(x,size_m=(filas,columnas))

    newrow = np.array(list(map(aux_func,new_coors[1:,:].T)))
    new_coors = np.vstack([new_coors, newrow])

    return sp.COO(new_coors,np.ones(new_coors.shape[1]),data_shape+(m,))


def add_independient_to_w(W):
    W_=W.todense()
    W_=np.append(W_, np.ones(W.shape[:-1]).reshape(W.shape[:-1]+(1,)), axis=3)
    return sp.COO(W_)


def create_W_continuo(filas,columnas,data_shape,data_size):
    ### W must have the size like a vector for each point in data

    m=filas*columnas ## number of covariates 
    new_coors=np.indices(data_shape,).reshape((3,data_size))
    aux_func = lambda x : get_flatten_pos(x,size_m=(filas,columnas))
    positions = np.array(list(map(aux_func,new_coors[1:,:].T))) / m
    new_coors2=np.zeros((4,new_coors.shape[1]*2))
    new_coors2[:3,::2]=new_coors
    new_coors2[:3,1::2]=new_coors
    new_coors2[-1,1::2]=1

    newrow = np.ones(2*len(positions))
    newrow[::2]=positions

    return sp.COO(new_coors2.astype(int),newrow,shape=data_shape+(2,))



###################################
## heuristic function

def heuristic_rule(data,thetha,W):
    
    return (W @ thetha)


##################################



def likelihood_cell_time(time,pos,thetha,data,W,likelihood_func):
    x,y=pos

    return likelihood_func(data[time,x,y]+1,thetha,W[time,x,y])
    


def get_best_change_cell_time(inputs):

    time,pos,thetha,data,neighborhood,W,likelihood_func = inputs

    eval_like=[]
    for pos in neighborhood[pos]:
        eval_like.append(likelihood_cell_time(time,pos,thetha,data,W,likelihood_func))

    return eval_like.index(min(eval_like))


def get_best_shift(thetha,data,neighborhood,W,likelihood_func):

    coding=[]
    for pos in list(neighborhood.keys()):

        try:
            index_data=data[:,pos[0],pos[1]].coords.flatten()
        except:
            index_data=np.where(data[:,pos[0],pos[1]]>0)[0]
        for time in index_data:
            code = [get_best_change_cell_time([time,pos,thetha,data,neighborhood,W,likelihood_func])] * int(data[:,pos[0],pos[1]][time])
            coding+=code

    return tuple(coding)

# def get_best_shift(thetha,data,neighborhood,W,likelihood_func):

#     coding=[]
#     for pos in list(neighborhood.keys()):

#         try:
#             index_data=data[:,pos[0],pos[1]].coords.flatten()
#         except:
#             index_data=np.where(data[:,pos[0],pos[1]]>0)[0]

#         inputs_=[[x,pos,thetha,
#                 data,neighborhood,
#                 W,likelihood_func] for x in index_data]

#         results = pool.map(get_best_change_cell_time, inputs_)

#         for idt,time in enumerate(index_data):
#             code= [results[idt]]*int(data[:,pos[0],pos[1]][time])
#             coding+=code

#     return tuple(coding)

def get_no_shift(data,neighborhood):
    coding=[]
    for pos in list(neighborhood.keys()):
        try:
            index_data=data[:,pos[0],pos[1]].coords.flatten()
        except:
            index_data=np.where(data[:,pos[0],pos[1]]>0)[0]
        for time in index_data:
            code = [(neighborhood[pos]).index(pos)] * int(data[:,pos[0],pos[1]][time])
            coding+=code
    return tuple(coding)


def do_shift(shift,data,neighborhood):

    try :
        new_data=sp.DOK(data.shape)
    except:
        new_data=np.zeros_like(data)

    total_events=0

    #####
    for pos in list(neighborhood.keys()):
        events_in_pos= int(data[:,pos[0],pos[1]].sum())

        new_pos=shift[total_events:total_events+events_in_pos]

        total_events+=events_in_pos

        index_data=np.where(data[:,pos[0],pos[1]]>0)[0]

        evento=0
        for time in index_data:
            for _ in range(int(data[:,pos[0],pos[1]][time])):
                new_x,new_y=neighborhood[pos][new_pos[evento]]
                new_data[time,new_x,new_y]+=1
                evento+=1

    return new_data


############# 

def BacktrackingLineSearch(alpha,xk,pk,f,gradf,rho=0.8,c=0.3):
  while f(xk+alpha*pk)>=f(xk)+c*alpha*(gradf(xk)@pk):
    alpha*=rho
    # print(f'paso={alpha},iteración={count}'
  return alpha




###### Prepare data

def transform_scale(vect):

    x=[vect.min(),vect.max()]
    y=[0,1]
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    return polynomial(vect)


def time_points_to_data_sparse(time_points,neighborhood,filas,columnas):
    time_points.coords[0] = transform_scale(time_points.coords[0])
    time_points.coords[1] = transform_scale(time_points.coords[1])

    data_posiciones=get_position(time_points.coords,(filas,columnas))
    min_year=time_points.times_datetime().min().year
    days_step=[(i.year-min_year)*365+i.timetuple().tm_yday for i in time_points.times_datetime()]

    dict_change={}
    for idx,i in enumerate(sorted(set(days_step))):
        dict_change[i]=idx

    time_steps=max(dict_change.values())

    days_step=[dict_change[i] for i in days_step]

    data_procc=np.concatenate((np.array(days_step).reshape(1,data_posiciones.shape[1]),data_posiciones))

    #############################################
    data_procc2=np.zeros((time_steps+1,filas,columnas))

    for t in range(time_steps+1):
        masked_time=data_procc.T[data_procc[0]==t].T

        temp_count_ev=np.zeros_like(data_procc2[t])

        for pos in neighborhood.keys():

            mask_pos=(masked_time[1:,:].T == np.array(pos)).sum(axis=1) == 2


            temp_count_ev[pos]=len(masked_time.T[mask_pos])

        data_procc2[t]=temp_count_ev
    #############################################
    data_sparse = sp.COO(data_procc2)

    return data_sparse


def gen_data_from_sparse(data_sparse,psi,neighborhood):

    to_move=np.random.binomial(data_sparse.data.astype(int),psi)
    to_move=sp.COO(data_sparse.coords,to_move,data_sparse.shape)

    moved=np.zeros(data_sparse.shape)
    for events,original_pos in zip(to_move.data,data_sparse.coords.T):
        o_pos=tuple(original_pos[1:])
        for event in range(events):
            new_pos=neighborhood[o_pos][np.random.choice(len(neighborhood[o_pos]))]
            moved[original_pos[0],new_pos[0],new_pos[1]]+=1

    moved = sp.COO(moved)

    data_moved=data_sparse-to_move+moved

    return data_moved


def gen_data_from_T0(psi,neighborhood,T,T0,total_days,mean_daily):
    data=np.zeros((total_days,)+T0.shape)

    events_by_time=np.random.poisson(mean_daily,total_days)

    # total_days
    for i in range(total_days):
        if i==0:
            compare=T0
        else:
            compare=data[max(0,i-T):i].mean(axis=0)

        hot_spot=np.unravel_index(compare.argmax(), compare.shape)

        for event in range(events_by_time[i]):

            rand_=np.random.rand()
            if rand_ < psi:
                new_pos=neighborhood[hot_spot][np.random.choice(len(neighborhood[hot_spot]))]
            else:
                new_pos=hot_spot

            data[i,new_pos[0],new_pos[1]]+=1
        
    return  sp.COO(data)
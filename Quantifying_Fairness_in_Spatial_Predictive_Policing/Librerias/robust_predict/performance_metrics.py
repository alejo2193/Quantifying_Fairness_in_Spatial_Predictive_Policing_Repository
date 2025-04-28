import numpy as np
import sparse as sp

from scipy.linalg import norm

def get_break_points(min,max,bins=3):
    return np.linspace(min,max,bins+1)


def mask_gen(data,break_points):
    mask_sqc=[]

    for i in range(len(break_points)-1):

        if i!=len(break_points)-2:

            mask = (data >= break_points [i]) & (data < break_points [i+1])
        
        else:
            mask = (data >= break_points [i]) & (data <= break_points [i+1])
        
        mask_sqc.append(mask)

    return mask_sqc

def rates_bins(data,mask_sqc):

    return np.nan_to_num(np.array([data[mask].mean() for mask in mask_sqc]))


def norm_rates(v1,v2,ord=np.inf):
    return norm(v1-v2,ord)


def compare_rates_bins(real,pred,min=None,max=None,bins=3,ord=np.inf):
    if not min:
        min= real.min()

    if not max:
        max= real.max()

    break_points=get_break_points(min,max,bins=bins)
    masks=mask_gen(real,break_points)

    rates_real=rates_bins(real,masks)
    rates_pred=rates_bins(pred,masks)

    return norm_rates(rates_real,rates_pred,ord)




def normalize_to_rates(data_sparse):
    return (data_sparse/data_sparse.sum(axis=[1,2],keepdims=True)).todense()




#######################  PAI & PEI

def rank_cells(alpha,rates_map):
    indices = np.argsort(rates_map.flatten())[::-1]
    n_cells=1
    while sum(rates_map.flatten()[indices[:n_cells]])< alpha:
        n_cells+=1
    if sum(rates_map.flatten()[indices[:n_cells]]) > alpha and n_cells>1:
        n_cells-=1

    cells_index=indices[:n_cells]

    return cells_index


def PAI_one_time(alpha,real,hot_pred):
    
    cells_index=rank_cells(alpha,hot_pred)

    n=real.flatten()[cells_index].sum()
    N=real.sum()
    return (n/N)/(len(cells_index)/real.size)

def PAI(alpha,real,hot_pred):

    values=[]
    for i in range(len(real)):
        values.append(PAI_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values)


def PEI_one_time(alpha,real,hot_pred):
    
    cells_index_pred=rank_cells(alpha,hot_pred)
    n=real.flatten()[cells_index_pred].sum()

    cells_index_real=rank_cells(alpha,real)
    n_ast=real.flatten()[cells_index_real].sum()

    return (n/len(cells_index_pred))/(n_ast/len(cells_index_real))

def PEI(alpha,real,hot_pred):

    values=[]
    for i in range(len(real)):
        values.append(PEI_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values)

    
def PEI_ast_one_time(alpha,real,hot_pred):
    
    cells_index_pred=rank_cells(alpha,hot_pred)
    n=real.flatten()[cells_index_pred].sum()

    
    n_ast=np.sort(real,axis=None)[::-1][:len(cells_index_pred)].sum()

    return n/n_ast


def PEI_ast(alpha,real,hot_pred):

    values=[]
    for i in range(len(real)):
        values.append(PEI_ast_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values)


######################### weighted travel

def get_Weighted_travel(res_opt,M_intensity):
    return sum([M_intensity[int(i[1].split(",")[0]),
                int(i[1].split(",")[1])
                ]*i[2] for i in res_opt])

def Weighted_travel(res_opt,intensity):

    values=[]
    for i in range(len(intensity)):
        values.append(get_Weighted_travel(res_opt,intensity[i]))

    return np.array(values)
import numpy as np
import pandas as pd
from sklearn import metrics

#############################################################################################################################

#Benefit functions

## benefit or utility for group (protected or no) from prediction
def f_benefit(mask_protected,real_events,intensity_matrix):

    assert(mask_protected.shape == real_events.shape), f"sizes protected and real events not same"
    assert(mask_protected.shape == intensity_matrix.shape), f"sizes protected and intensity not same"
    assert(real_events.shape == intensity_matrix.shape), f"sizes real events and intensity not same"

    if real_events.sum() != 1:
        if real_events.sum() != 0:
            real_events=real_events/real_events.sum()

    if intensity_matrix.sum() != 1:
        if intensity_matrix.sum() != 0:
            intensity_matrix=intensity_matrix/intensity_matrix.sum()

    return abs(intensity_matrix[mask_protected]-real_events[mask_protected]).mean()


## benefit for k cells with most intensity
def f_benefit_k_most(mask_protected,real_events,intensity_matrix,k):

    assert(mask_protected.shape == real_events.shape), f"sizes protected and real events not same"
    assert(mask_protected.shape == intensity_matrix.shape), f"sizes protected and intensity not same"
    assert(real_events.shape == intensity_matrix.shape), f"sizes real events and intensity not same"

    if real_events.sum() != 1:
        real_events=real_events/real_events.sum()

    if intensity_matrix.sum() != 1:
        intensity_matrix=intensity_matrix/intensity_matrix.sum()

    top_k=sum(real_events[mask_protected]!=0)
    if k<=top_k:
        None
    else:
        k=top_k
    M=np.array([intensity_matrix[mask_protected],real_events[mask_protected]])
    pandas_top=pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).head(k)>0

    tp=len(pandas_top[(pandas_top[0]==True) & (pandas_top[1]==True)])  
    tn=len(pandas_top[(pandas_top[0]==False) & (pandas_top[1]==False)])  
    fp=len(pandas_top[(pandas_top[0]==True) & (pandas_top[1]==False)])  
    fn=len(pandas_top[(pandas_top[0]==False) & (pandas_top[1]==True)])  

    confusion_matrix_m=np.array([[tp, fp], [fn, tn]])
    confusion_matrix_m
    
    try:
        precision=confusion_matrix_m[0][0]/(confusion_matrix_m[0][0]+confusion_matrix_m[0][1])
        accuracy=(confusion_matrix_m[0][0]+confusion_matrix_m[1][1])/(confusion_matrix_m[0][0]+confusion_matrix_m[0][1]+confusion_matrix_m[1][0]+confusion_matrix_m[1][1])
    except:
        precision=None
        accuracy=None

    return confusion_matrix_m, precision, accuracy
    #return abs(pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).head(k).sum(axis=1)).mean()
    #return abs(pd.DataFrame(M).T.sort_values(1,ascending=True).reset_index(drop=True).head(k).sum(axis=1)/pd.DataFrame(M).T.sort_values(1,ascending=True).reset_index(drop=True).head(k)[1]).mean()
    

## benefit for k-th cell with most intensity
def f_benefit_k_ind(mask_protected,real_events,intensity_matrix,k,abs_=False,under_over=None):

    if real_events.sum() != 1:
        real_events=real_events/real_events.sum()

    if intensity_matrix.sum() != 1:
        intensity_matrix=intensity_matrix/intensity_matrix.sum()

    M=np.array([intensity_matrix[mask_protected],-real_events[mask_protected]])

    R=pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).loc[k].sum()
    if abs_ and under_over==None:
        return abs(R)
    elif ~abs_ and under_over==None:
        return R
    elif under_over=="under":
        return max(R,0)
    elif under_over=="over":
        return max(-R,0)
    else:
        return Exception()

#############################################################################################################################

# Fairness functions lower is better  (just 1 protected variable)

#AD Absolute difference
def Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit=f_benefit,ben_kwgs={}):
    try:
        return abs(f_benefit(mask_protected,real_events,intensity_matrix,**ben_kwgs)-f_benefit(~mask_protected,real_events,intensity_matrix,**ben_kwgs))
    except Exception as e:
        return e

## value_unfairness
def value_unfairness(mask_protected,real_events,intensity_matrix):
    v_u={}
    v_u["val"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"abs_":False}) for k in range(0,min(mask_protected.sum(),(~mask_protected).sum()))]).mean()
    v_u["abs"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"abs_":True}) for k in range(0,min(mask_protected.sum(),(~mask_protected).sum()))]).mean()
    v_u["under"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"under_over":"under"}) for k in range(0,min(mask_protected.sum(),(~mask_protected).sum()))]).mean()
    v_u["over"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"under_over":"over"}) for k in range(0,min(mask_protected.sum(),(~mask_protected).sum()))]).mean()

    return v_u

# Fairness functions lower is better  (more protected variables)

## square of difference between banefict function of two groups
def square_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        return (f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))**2
    except Exception as e:
        return e

## abs of difference between banefict function of two groups
def abs_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        return abs(f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))
    except Exception as e:
        return e

## Mean of difference between banefict function of all pair of groups
def variance(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        return np.array([square_diff_benefit(dict_masks[i],dict_masks[j],f_benefit,real_events,intensity_matrix) for i in dict_masks.keys() for j in dict_masks.keys()]).mean()
    except Exception as e:
        return e

## Max difference between benefict function of all pair of groups
def MM(dict_masks,f_benefit,real_events,intensity_matrix,k_p_up,measure="Diff"):
    try:
        R=[f_benefit_k_most(dict_masks[i],real_events,intensity_matrix,k_p_up[i]) for i in dict_masks]
        if measure == "Diff":
            return max(R)-min(R)
        if measure == "Ratio":
            return min(R)/max(R) #(higher is better)

    except Exception as e:
        return e

## gini coefficient
def gini_coeff(dict_masks,f_benefit,real_events,intensity_matrix,k_p_up):
    try:
        R=np.array([f_benefit_k_most(dict_masks[i],real_events,intensity_matrix,k_p_up[i]) for i in dict_masks])
        # R=1-R/R.sum()
        den=2*len(R)*sum(R)
        G=abs(R.reshape(len(R),1)-R.reshape(1,len(R))).sum()/den
        return G
    except Exception as e:
        return e


## jain index (higher is better)
def jain_index(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        R=np.array([f_benefit(dict_masks[i],real_events,intensity_matrix) for i in dict_masks])
        # R=1-R/R.sum()
        den=len(R)*sum(R**2)
        J=(R.sum())**1/den
        return J
    except Exception as e:
        return e


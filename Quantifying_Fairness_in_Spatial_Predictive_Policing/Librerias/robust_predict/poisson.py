import numpy as np
import sparse as sp
from scipy.special import factorial
from scipy.optimize import minimize
import cvxpy as cp


#### likelihood 

def compute_likelihood(data,thetha,W):
    try:
        return (data*(W @ thetha)-np.exp(W @ thetha)-np.log(factorial(data.todense()))).sum()
    except:
        return (data*(W @ thetha)-np.exp(W @ thetha)-np.log(factorial(data))).sum()


def jac_like (data,thetha,W):
    jac_= np.array([(W [:,:,:,i] * (data-np.exp(W @ thetha))).sum() for i in range(len(thetha))])
    return jac_/np.linalg.norm(jac_)

### func to optimize thetha
def to_opt_theta(data,thetha,W):
    
    try: return np.linalg.norm( (data.sum(axis=0)- np.exp(W @ thetha).sum(axis=0)).todense())
    except: return np.linalg.norm( data.sum(axis=0)- np.exp(W @ thetha).sum(axis=0))


#compute thetha
def best_theta(t_0,data,W):

    # bnds = [(0, None) for i in range(t_0.size)]

    X=minimize(lambda x : to_opt_theta(data,x,W),t_0,
               method="Powell"
            #    bounds=bnds,
               )
    print(X["message"])

    return X.x

##### method cvxpy
def likelihood_2d(data,thetha,W):
    log__= -np.log(factorial(data.reshape((data.shape[0],data.shape[1]*data.shape[2])).todense())).sum()
    tempL = data.todense().reshape((data.size,)) @ (W.todense().reshape((data.size,thetha.size)) @ thetha) - cp.sum(cp.exp(W.todense().reshape((data.size,thetha.size)) @ thetha)) +log__
    return tempL
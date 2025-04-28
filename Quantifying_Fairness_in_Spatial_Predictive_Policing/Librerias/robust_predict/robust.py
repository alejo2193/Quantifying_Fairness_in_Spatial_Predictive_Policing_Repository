from .poisson import *
from .utils import *
import cvxpy as cp
import pickle

class poissonProblemVector:
    def __init__(self, numFeatures,solver="SCS"):
        self.numFeatures = numFeatures
        self.delta = cp.Variable()
        self.theta = cp.Variable(numFeatures,nonneg=False)
        self.constraints = []
        self.solver=solver

        # define objective
        self.obj = cp.Maximize(self.delta)
        self.formulation = cp.Problem(self.obj, self.constraints)

    def addConstraint(self, data,W):
        """
        Adds a constraint based on the current best response of the attacker.
        :param updatedData: Set of manipulations chosen by the attacker at current time step
        :return:
        """
        cons = self.delta
        cons -= likelihood_2d(data,self.theta,W)
        

        self.constraints.append(cons <= 0)

    def redefineProblem(self):
        """
        Updates the problem with the latest set of constraints
        :return: _
        """
        self.formulation = cp.Problem(self.obj, self.constraints)

    def solve(self,new_solver=None):
        """
        Solve the current optimization problem. If SCS returns an error, try without an explicit solver.
        :return: _
        """
        print("Attempting to solve problem instance with {} constraints".format(len(self.constraints)))
        
        try:
            if new_solver:
                self.formulation.solve(solver=new_solver)    
            else:
                self.formulation.solve(solver=self.solver)    
            # self.formulation.solve(solver='ECOS')
        except:
            print("error in solver, trying another")
            self.formulation.solve()
        print(self.formulation.status)

    def step_opt(self,data,W):

        self.addConstraint(data,W)
        self.redefineProblem()
        self.solve()
        return np.array(list(self.theta.value))
    
    def save_state(self,path_file):
        with open(path_file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def RSALA(data,W,neighborhood,likelihood_func=compute_likelihood,epsilon=1e-5,solver="SCS",max_iter=20,save=None):
    optProb = poissonProblemVector(W.shape[-1],solver=solver)
    # add constraint and redefine problem
    optProb.addConstraint(data,W)
    optProb.redefineProblem()
    optProb.solve()

    thetha_new = np.array(list(optProb.theta.value))

    gap=10
    iter_ = 0
    

    while abs(gap)> epsilon and iter_ <max_iter:
        iter_+=1
        thetha_current=thetha_new.copy()


        best_shift=get_best_shift(thetha_current,data,neighborhood,W,likelihood_func)
        update_data=do_shift(best_shift,data,neighborhood)

        try:
            thetha_new=optProb.step_opt(update_data,W)

            gap = compute_likelihood(update_data,thetha_new,W) - compute_likelihood(update_data,thetha_current,W)

            if save:
                optProb.save_state(save)
        except:
            print("Finish model in iteration ", iter_)

    return thetha_new, update_data


def ADGRAD (data,W,neighborhood,likelihood_func=compute_likelihood,epsilon=1e-5,solver="SCS",max_iter=50,save=None):

    optProb = poissonProblemVector(W.shape[-1],solver=solver)
    # add constraint and redefine problem
    optProb.addConstraint(data,W)
    optProb.redefineProblem()
    optProb.solve()

    thetha_new = np.array(list(optProb.theta.value))

    gap=10
    iter_ = 0

    alpha= BacktrackingLineSearch(100,thetha_new,
                                  -jac_like(data,thetha_new,W),
                                  lambda x: -compute_likelihood(data,x,W),
                                  lambda x: -jac_like(data,x,W),
                                  rho=0.8,c=0.3)

    while abs(gap)> epsilon and iter_ <=max_iter:
        iter_+=1

        thetha_current=thetha_new.copy()


        best_shift=get_best_shift(thetha_current,data,neighborhood,W,likelihood_func)
        update_data=do_shift(best_shift,data,neighborhood)

        

        thetha_new= thetha_current + alpha* jac_like(update_data,thetha_current,W)

        alpha= BacktrackingLineSearch(alpha,thetha_new,
                                  -jac_like(data,thetha_new,W),
                                  lambda x: -compute_likelihood(update_data,x,W),
                                  lambda x: -jac_like(update_data,x,W),
                                  rho=0.8,c=0.3)

        gap = compute_likelihood(update_data,thetha_new,W) - compute_likelihood(update_data,thetha_current,W)
        print("Iteration: ",iter_," gap: ", gap, "alpha: ", alpha)

        if save:
            optProb.save_state(save)

    return thetha_new, update_data


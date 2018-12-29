#!/usr/bin/env python
from scipy.optimize import minimize
from scipy import *

def fun(x):
    return (x[0]-2)**2+(x[1])**2
def grad(x):
    return array([2*(x[0]-2),2*x[1]])
def fun_grad(x):
    return (x[0]-2)**2+(x[1])**2,array([2*(x[0]-2),2*x[1]])
if __name__=="__main__":
	print minimize(fun, array([-1,1]), method='BFGS', jac=grad).nit
	print minimize(fun_grad, array([-1,1]), method='BFGS', jac=True).x

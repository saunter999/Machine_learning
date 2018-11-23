#!/usr/bin/env python
from scipy import *
from pylab import *
import csv
import numpy as np
from numpy import linalg as LA

class LRegres:
	"""
	This class supports (batch) Gradient descent/normal equation method to compute the Linear 
	Regression weights/coefficients for a given training dataset.  
	"""
	def __init__(self,n,m,data,num_iter,method):
	    self.n=int(n)
	    self.m=int(m)
	    self.data=data
	    self.num_iter=num_iter
	    self.X=data[:,:-1]
	    self.Y=zeros((m,1))
	    self.Y[:,0]=data[:,-1]
	    self.method=method
	    self.theta=zeros((n+1,1))
#	    self.theta[0,0]=-1;self.theta[1,0]=2
#	    print self.theta
	    if self.method=='GD':
	       self.alpha=0.01
	    self.X0=ones((m,1))   
	    self.X=np.concatenate((self.X0,self.X),axis=1)
	    self.mu=zeros((1,n+1))
	    self.sigma=zeros((1,n+1))

	def printinfo(self):
	    print "number of training examples:%.0f" % self.m
	    print "number of features:%.0f" % self.n
	    if self.method=='GD':
 	       print "method used to do linear regression: ", 'Gradient descent'
	       print 'number of iterations:%.0f ' % self.num_iter
	       print 'Learning rate:%.3f ' % self.alpha

	def featureNormalizing(self):
	    if (self.n!=1)and(self.method=='GD'):
	        self.mu[0,:]=np.mean(self.X,axis=0)
                self.X[:,1:]-=self.mu[0,1:]
 	        self.sigma[0,:]=np.std(self.X,axis=0)
                self.X[:,1:]/=self.sigma[0,1:]
#  		print self.mu,self.sigma
#   		print self.X
#		return [self.mu,self.sigma]

	def computeCost(self):
	    J=0.0
	    h=np.dot(self.X,self.theta)
	    dev=h-self.Y
	    J=1./(2.*self.m)*LA.norm(dev)**2
#	    print J
	    return J
	    
	def Gradientdescent(self):
#	    print self.theta,h.shape,(self.Y).shape
	    for i in range(self.num_iter):
	        h=np.dot(self.X,self.theta)
	        dev=h-self.Y
		for j in range(self.n+1):
 	            self.theta[j]-=self.alpha/self.m * np.dot(dev.transpose(),self.X[:,j])
	    
	    

        def findtheta(self):
	    if self.method=='GD':
	       self.featureNormalizing()
	       self.Gradientdescent()
	       return self.theta,self.mu,self.sigma





if __name__=="__main__":
	souf=open('ex1data1.txt','r')
	data=loadtxt(souf,delimiter=',')
	print "Training dataset has the shape of [%.0f %.0f]" % data.shape
	m=data.shape[0];n=data.shape[1]-1
	my_LR=LRegres(n,m,data,num_iter=2500,method='GD')
	my_LR.printinfo()
# 	my_LR.computeCost()
	theta=my_LR.findtheta()
	
	print theta
	

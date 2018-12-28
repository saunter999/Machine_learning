#!/usr/bin/env python
import numpy as np
#from numpy import linalg as LA
from scipy import *
from pylab import *
from copy import deepcopy as dp

class conn_cc:
    def __init__(self,struc,X,Y,activ_type):
    	"""
	Initialization of the network structure of the fully
	connected neural network
	"""
	self.struc=struc
	self.L=len(self.struc)
	self.X,self.Y=dp(X),dp(Y)
	self.activ=activ_type
	self.nsample=self.X.shape[0]
	self.W=[] ##weights of the whole network
	for i in range(1,self.L):
	    s_o=self.struc[i]
	    s_i=self.struc[i-1]	
	    self.W.append( np.random.rand(s_o,s_i+1)-0.5 ) ### added 1 due to the bias unit
	#print self.W

    def printinfo(self):
          print ("~~~~~~~~~~INFO of Fully connected neural network~~~~~~~~~~") 
	  print ("conn_cc Layer #:",self.L) 
	  print ("# of units across the network starting from the input layer:",self.struc)
	  print ("# of features or units of the input layer:",self.struc[0]) 
	  print ("# of units of the output layer:",self.struc[-1]) 
	  print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 

    def Forward_prop(self):
    	bias=ones((self.nsample,1))
        source=self.X
    	for i in range(self.L-1):
	    if i==0:
	       sb=np.concatenate( (bias,source),axis=1)
	    else:
	       sb=np.concatenate((bias,out),axis=1)
	    z=np.dot(sb,self.W[i].transpose())
	    print 'z=',z
	    out=self.activation(z,self.activ)
	    print("Forwarding propagating passing through layer",i+1)
	    print ("The shape of resulting output",out.shape)
	print out

    def activation(self,z,activ):
        if activ=='sigmoid':
	   return 1.0/(1.0+exp(-z))


       
        

if __name__=="__main__":
	nfeature=2
	nout=6
	nsample=20
	X=np.random.rand(nsample,nfeature)-0.5
	print X
 	Y=np.random.rand(nsample,nout)
        struc=[nfeature,6,9,nout]
	mycnn=conn_cc(struc,X,Y,'sigmoid')
	mycnn.printinfo()
	mycnn.Forward_prop()

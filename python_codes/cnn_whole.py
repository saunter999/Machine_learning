#!/usr/bin/env python
import numpy as np
#from numpy import linalg as LA
from scipy import *
from pylab import *
from copy import deepcopy as dp
import time

class conn_cc:
    def __init__(self,struc,X,Y,nout,activ_type):
    	"""
	Initialization of the network structure of the fully
	connected neural network
	X-input data,shape (nsample,nfeature)
	Y-labels,shape (nsample,1)
	"""
	self.struc=struc
	self.L=len(self.struc)
	self.X,self.Y,self.nout=dp(X),dp(Y),dp(nout)
	self.nsample=self.X.shape[0]
	self.Yvec=zeros((nsample,self.nout))
	self.activ=activ_type
	self.W=[] ##weights of the whole network
	for i in range(1,self.L):
	    s_o=self.struc[i]
	    s_i=self.struc[i-1]	
	    if i==1:
	      self.W.append(loadtxt('Theta1.txt'))
	    else:
	      self.W.append(loadtxt('Theta2.txt'))
	    #self.W.append( np.random.rand(s_o,s_i+1)) ### added 1 due to the bias unit
	#print (self.W)

    def printinfo(self):
          print ("~~~~~~~~~~INFO of Fully connected neural network~~~~~~~~~~") 
	  print ("conn_cc Layer #:",self.L) 
	  print ("# of units across the network starting from the input layer:",self.struc)
	  print ("# of features or units of the input layer:",self.struc[0]) 
	  print ("# of units of the output layer:",self.struc[-1]) 
	  print ("~~~~~~~~~~End of INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 


    def Train(self):
	return self.W


    def Predict(self,X):
        out=self.Forward_prop(X)
	return argmax(out,axis=1)+1

    def Forward_prop(self,X):
    	bias=ones((self.nsample,1))
        source=X
    	for i in range(self.L-1):
	    if i==0:
	       sb=np.concatenate( (bias,source),axis=1)
	    else:
	       sb=np.concatenate((bias,out),axis=1)
	    z=np.dot(sb,self.W[i].transpose())
	#    print ('z=',z)
	    out=self.activation(z,self.activ)
	#    print ('out=',out)
	    print("Forward propagating passing through layer",i+1)
	    print ("The shape of resulting output",out.shape)
	return out


    def activation(self,z,activ):
        if activ=='sigmoid':
	   return 1.0/(1.0+exp(-z))

    def costfunc(self,param,X):
	for idx,y in enumerate(self.Y):
	    self.Yvec[idx,int(y)-1]=1
#	    print y,self.Yvec[idx,:]
	h=self.Forward_prop(X)
	cost=-1.0/self.nsample*np.trace( np.dot(log(h),self.Yvec.transpose())+np.dot( log(1.0-h),(1.0-self.Yvec).transpose()) )
	#cost=0.0
	#h=self.Forward_prop(X)
	#for i in range(self.nsample):
	#    cost+=-1.0/self.nsample*( np.dot( log(h)[i,:],self.Yvec.transpose()[:,i])+np.dot( log(1.0-h)[i,:],(1.0-self.Yvec).transpose()[:,i]) )
        print ('cost=',cost) 

    def grad_eval(self):
    	print(1)


    def Back_prop(self):
        print(1)
     
    def score(self,X,Y):
        ypred=self.Predict(X)
	return sum(ypred==Y)/float(nsample) 
       
        

if __name__=="__main__":
        t1=time.time()
	X=loadtxt("Xinp.txt")
	Y=loadtxt("Yout.txt")
	nfeature=X.shape[1]
	nout=int( max(Y))
	nsample=X.shape[0]
	print("nfeature=",nfeature,"nout=",nout,"nsample",nsample)
        struc=[nfeature,25,nout]
	mycnn=conn_cc(struc,X,Y,nout,'sigmoid')
	mycnn.printinfo()
	myWgt=mycnn.Train()
	mycnn.costfunc(myWgt,X)
	ypred=mycnn.Predict(X)
	score=mycnn.score(X,Y)
	print('ypred=',ypred)
	print ("Accuracy=",score )
	t2=time.time()
	print('running time:',t2-t1)

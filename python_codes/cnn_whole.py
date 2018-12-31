#!/usr/bin/env python
import numpy as np
#from numpy import linalg as LA
from scipy import *
from pylab import *
from copy import deepcopy as dp
from scipy.optimize import minimize
import time

class conn_cc:
    def __init__(self,struc,X,Y,nout,tol,Nit,C=0.0,activ_type=None,initw=None):
    	"""
	Initialization of the network structure of the fully
	connected neural network
	struc-List consisting of # of neuron units across the neural network
	X-input data,shape (nsample,nfeature)
	Y-labels,shape (nsample,1)
	C-regularization parameter
	activ_type:activation function used in nn
	"""
	self.struc=struc
	self.L=len(self.struc)
	self.X,self.Y,self.nout,self.C,self.tol,self.Nit=dp(X),dp(Y),dp(nout),dp(C),dp(tol),dp(Nit)
	self.nsample=self.X.shape[0]
	self.Yvec=zeros((nsample,self.nout))
	for idx,y in enumerate(self.Y):
	    self.Yvec[idx,int(y)-1]=1

	self.activ,self.initw = activ_type,initw
	self.W=[] ##weights of the whole network
	epsilon_init = 0.12
	if self.initw=='Random':
	    print("Random initializing weights across the neural network")
	    for i in range(1,self.L):
	      s_o=self.struc[i]
	      s_i=self.struc[i-1]	
	      self.W.append( np.random.rand(s_o,s_i+1)*2.0*epsilon_init-epsilon_init) ### added 1 due to the bias unit
	else:
	    print("Preloading weights across the neural network")
	    self.W.append(loadtxt('Theta1.txt'))
	    self.W.append(loadtxt('Theta2.txt'))

    def printinfo(self):
          print ("~~~~~~~~~~INFO of Fully connected neural network~~~~~~~~~~") 
	  print ("conn_cc Layer #:",self.L) 
	  print ("# of units across the network starting from the input layer:",self.struc)
	  print ("# of features or units of the input layer:",self.struc[0]) 
	  print ("# of units of the output layer:",self.struc[-1]) 
	  print ("~~~~~~~~~~End of INFO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~") 


    def unrollWeight(self,wgt):
    	nn_params=[w.reshape(-1,) for w in wgt]
	nn_params=np.concatenate(nn_params)
	
	return nn_params

    def rollingWeight(self,params):
        n=0
	Weight=[]
	for i in range(1,self.L):
	  s_o=self.struc[i]
	  s_i=self.struc[i-1]	
	  wlen=s_o*(s_i+1)
	  Weight.append(params[n:n+wlen].reshape(s_o,s_i+1))
	  n+=wlen
	return Weight

    def Train(self,method):
	if method=='BFGS':
	   nn_params=self.unrollWeight(self.W)
	   res=minimize(self.costfunc,nn_params,method='BFGS',jac=self.grad_eval,\
		        tol=self.tol,options={'disp':True,'maxiter':self.Nit})
	   self.W=self.rollingWeight(res.x)
	   print("Optimization status",res.message)
	   print("# of iterations=",res.nit)

    def grad_eval(self,params):
	if type(params[0])!=np.ndarray:
	   print("Rolling back params into array to calculate gradient")
	   weight=self.rollingWeight(params)
	else:
	   print("No need to roll back params into array to calculate gradient")
	   weight=params
	w_grad=[]
	for w in weight:
	    w_grad.append( zeros(w.shape))
	#for i in range(self.nsample):
	for i in range(1):
	  ##1.performing forward first
          source=X[i,:]
	  a=[source]
	  ab=[np.concatenate( (array([1]),source),axis=0)]
	  for j in range(self.L-1): 
	    if j==0:
	       sb=np.concatenate( (array([1]),source),axis=0)
	    else:
	       sb=np.concatenate((array([1]),out),axis=0)
	    z=np.dot(sb,weight[j].transpose())
	    out=self.activation(z,self.activ)
	    a.append(out)
	    ab.append(np.concatenate( (array([1]),out),axis=0))
	  ##2.performing backward to calculate delta
	  Delta=[]
	  for j in range(self.L,1,-1):
	      if j==self.L:  
	          dta=a[-1]-self.Yvec[i]    #print a[-1].shape,self.Yvec[i].shape#delta=a[-1]-self.Yvec
	      else:
	          dta=np.dot(weight[j-1].transpose(),Delta[-1])*ab[j-1]*(1.0-ab[j-1]) ##assuming sigmoid activation func  here
		  dta=dta[1:]
              Delta.append(dta)
	  Delta=Delta[::-1]
	  print Delta

    def costfunc(self,params):
    	cost=0.0
	if type(params[0])!=np.ndarray:
	   print("Rolling back params to into array calculate costs")
	   weight=self.rollingWeight(params)
	else:
	   print("No need to roll back params into array to calculate costs")
	   weight=params
	h=self.Forward_prop(weight,self.X)
	cost=-1.0/self.nsample*np.trace( np.dot(log(h),self.Yvec.transpose())+np.dot( log(1.0-h),(1.0-self.Yvec).transpose()) )

	#adding regularization part 
	for w in weight:
	    cost+=self.C/(2.0*self.nsample)*sum(w[:,1:]**2)
        print ('cost=',cost) 

    def Forward_prop(self,weight,X):
    	bias=ones((self.nsample,1))
        source=X
    	for i in range(self.L-1):
	    if i==0:
	       sb=np.concatenate( (bias,source),axis=1)
	    else:
	       sb=np.concatenate((bias,out),axis=1)
	    z=np.dot(sb,weight[i].transpose())
	    out=self.activation(z,self.activ)
	    print("Forward propagating passing through layer",i+1)
	    print ("The shape of resulting output",out.shape)
	return out

    def Predict(self,params,X):
	if type(params[0])!=np.ndarray:
	   print("Rolling back params into array to predict label")
	   weight=self.rollingWeight(params)
	else:
	   print("No need to roll back params into array to predict label")
	   weight=params
        out=self.Forward_prop(weight,X)
	return argmax(out,axis=1)+1

     
    def score(self,weight,X,Y):
        ypred=self.Predict(weight,X)
	return sum(ypred==Y)/float(nsample) 
       
        
    def activation(self,z,activ):
        if activ=='sigmoid':
	   return 1.0/(1.0+exp(-z))

if __name__=="__main__":
        t1=time.time()
	X=loadtxt("Xinp.txt")
	Y=loadtxt("Yout.txt")
	nfeature=X.shape[1]
	nout=int( max(Y))
	nsample=X.shape[0]
	print("nfeature=",nfeature,"nout=",nout,"nsample",nsample)
        struc=[nfeature,25,19,16,nout]
	#mycnn=conn_cc(struc,X,Y,nout,0.01,1,C=1.0,activ_type='sigmoid')
	mycnn=conn_cc(struc,X,Y,nout,0.01,1,C=1.0,activ_type='sigmoid',initw='Random')
	mycnn.printinfo()
	#exit()
	mtd='BFGS'
	mywgt=mycnn.Train(method=mtd)
	mycnn.costfunc(mywgt)
	score=mycnn.score(mywgt,X,Y)
	print ("Accuracy=",score )
	t2=time.time()
	print('running time:',t2-t1)

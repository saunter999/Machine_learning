#!/usr/bin/env python
import numpy as np
from numpy import linalg as LA
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
	nout- number of labels in the output
	tol- tolerance in the minimization process
	Nit- maximum number of iteration in the minimization process
	C-regularization parameter
	activ_type:activation function used in nn
	initw: way of initializing weights across nn
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
	if self.initw=='Random':
	    print("Random initializing weights across the neural network")
	    for i in range(1,self.L):
	      s_o=self.struc[i]
	      s_i=self.struc[i-1]	
	      epsilon_init = sqrt(6./(s_i+s_o))
	      print("epsilon_init=",epsilon_init)
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
	  print("tol=",self.tol,'Nit=',self.Nit,'Regularization parameter=',self.C)
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

    def Train(self,epochs):
           alpha=0.1
	   self.W=self.unrollWeight(self.W)
	   batch_size=self.nsample
	   Nb=int(self.nsample/batch_size)
	   print(Nb)
	   for i in range(epochs):
	      print ("epochs:",i)
	      randind=np.random.permutation(self.Y.shape[0])
	      self.X=self.X[ randind ]
	      self.Y=self.Y[ randind ]
	      for j in range(Nb):
	          grad=self.grad_eval(self.W,j,batch_size)
		  print 'grad norm=',LA.norm(grad)
 	          self.W+=-alpha*grad
	      #cost=self.costfunc(self.W)
	      #print(i,cost)
#	   self.W=self.rollingWeight(self.W)
	   return self.W

    def grad_eval(self,params,ie,bsize):
	if type(params[0])!=np.ndarray:
	   weight=self.rollingWeight(params)
	else:
	   weight=params
	w_grad=0.0
	print('~~~~~~~evaluating gradient using backprop~~~~~~~~~~~')
	for i in range(ie*bsize,(ie+1)*bsize):
	  ##1.performing forwardprop first
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
	  ##2.performing backprop to calculate delta
	  Delta=[]
	  for j in range(self.L,1,-1):
	      if j==self.L:  
	          dta=a[-1]-self.Yvec[i]    
	      else:
	          dta=np.dot(weight[j-1].transpose(),Delta[-1])*ab[j-1]*(1.0-ab[j-1]) ##assuming sigmoid activation func  here
		  dta=dta[1:]
              Delta.append(dta)
	  Delta=Delta[::-1]
	  ##3.tensor product between delta and ab(activation with bias) to get gradient
	  w_grad_i=[]
	  for j in range(len(Delta)):
             res=np.outer(Delta[j],ab[j])
	     res[:,1:]+=self.C*weight[j][:,1:]/bsize  #Adding regularization part
	     w_grad_i.append(res)
	  w_grad+=self.unrollWeight(w_grad_i)
	w_grad*=1./bsize
	print('~~~~~~~finishing gradient using backprop~~~~~~~~~~~')
	return w_grad


    def costfunc(self,params):
    	cost=0.0
	if type(params[0])!=np.ndarray:
	   weight=self.rollingWeight(params)
	else:
	   weight=params
	h=self.Forward_prop(weight,self.X)
	cost=-1.0/self.nsample*np.trace( np.dot(log(h),self.Yvec.transpose())+np.dot( log(1.0-h),(1.0-self.Yvec).transpose()) )
	#adding regularization part 
	for w in weight:
	    cost+=self.C/(2.0*self.nsample)*sum(w[:,1:]**2)
	print ('cost=',cost) 
	return np.asscalar(cost) 

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
	return out

    def Predict(self,params,X):
	if type(params[0])!=np.ndarray:
	   weight=self.rollingWeight(params)
	else:
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
        struc=[nfeature,25,20,nout]
	#mycnn=conn_cc(struc,X,Y,nout,0.01,1,C=1.0,activ_type='sigmoid')
 	epols=range(40,60,20)
	res=[]
	for epochs in epols:
	  print("# of epochs:",epochs)
	  mycnn=conn_cc(struc,X,Y,nout,1e-4,50,C=1.0,activ_type='sigmoid',initw='Random')
	  mycnn.printinfo()
  	  score=mycnn.score(mycnn.W,X,Y)
	  print ("Before training,Accuracy=",score )
	  mywgt=mycnn.Train(epochs)
	  score=mycnn.score(mywgt,X,Y)
	  res.append(score)
	  print ("After training,Accuracy=",score )
	  t2=time.time()
	  print('running time:',t2-t1)
	plot(epols,res,'o-')
	xlabel("epochs",size='large')
	ylabel("Accuracy",size='large')
	savefig("Acc.png")
	show()

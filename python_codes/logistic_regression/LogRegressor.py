from scipy import *
from pylab import *
import numpy as np
from numpy import linalg as LA
from copy import deepcopy as dp
from scipy.optimize import minimize

class LogRegress:

  def __init__(self,X,Y,C=0.0,batch_size=1,nfflag=True,costflag=True):
    """
    X is the training data with shape(nsample,nfeatures); 
    Y is the corresponding label with shape (nsample,1).
    Nit-maximum number of iterations
    tol-tolerance
    alpha-learning rate
    C-regularization parameter (L2 norm)
    nfflag: flag for normalizing features
    costflag: flag for computing cost or not
    """
    self.X, self.Y ,self.C, self.batch_size, self.nfflag, self.costflag= dp(X),dp(Y),dp(C),dp(batch_size),dp(nfflag),dp(costflag)
    print("~~~~~~~~~~~hyperparameters info~~~~~~~~~~~~")
#    print("Maximum_iter =",self.Nit)
#    print("tolerance=",self.tol)
#    print("learning rate for customized minimization algorthim=",self.alpha)
    print("regularization parameter=",self.C)
    print("Flag for normalizing features:",self.nfflag)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    if nfflag: self.Normalfeature()
    self.Y=self.Y.reshape(-1,1)
    self.nsample,self.nfeature=self.X.shape[0],self.X.shape[1]
    bias=ones((self.nsample,1))
    self.Xb=np.concatenate( (bias,self.X),axis=1)
    self.theta=zeros( (self.nfeature+1,1) )
    self.cost=[]
    
  def training(self,Nit,tol,alpha,method):
      mini_param=[Nit,tol,alpha]
      methodls=['SGD','BFGS','Newton-CG']
      if method not in methodls: raise Exception("Minimization method ("+method+") is not available.")

      if method=='SGD':
         self.theta=self.SGD(self.theta,self.grad_eval,mini_param,self.batch_size)

      if method=='BFGS':
         res=minimize(self.costfunction,(self.theta).reshape(-1,),method='BFGS',jac=self.grad_eval,\
	 tol=self.tol,options={'disp':True,'maxiter':self.Nit})
	 self.theta=res.x.reshape(-1,1)
	 print("Optimization status",res.message)
	 print("# of iterations=",res.nit)

      if method=='Newton-CG':
         res=minimize(self.costfunction,(self.theta).reshape(-1,),method='Newton-CG',\
	 jac=self.grad_eval,tol=self.tol,options={'disp':True,'maxiter':self.Nit}) 
	 self.theta=res.x.reshape(-1,1)
	 print("Optimization status",res.message)
	 print("# of iterations=",res.nit)

  def grad_eval(self,theta,ib,bsize):
	  batch=range(ib*bsize,(ib+1)*bsize)     
          theta=theta.reshape(-1,1)
          linarg=np.dot(self.Xb[batch],theta)
          h=array([self.logisticfunc(z) for z in linarg]).reshape(-1,1)
          grad=1.0/bsize*np.dot( self.Xb[batch].transpose(),h-self.Y[batch] )
          grad[1:]+= self.C/self.nsample*theta[1:] ##Adding regularized part into the gradient
      	  return grad.reshape(-1,)
        
  def SGD(self,theta,gradfunc,param,bsize):
      Nit,tol,learning_rate=param
      Nb=int(self.nsample/bsize)
      for i in range(Nit):
        print (" Iterations/Epochs:",i)
	randind=np.random.permutation(self.nsample)
	self.X=self.X[ randind ]
	self.Y=self.Y[ randind ]
        bias=ones((self.nsample,1))
    	self.Xb=np.concatenate( (bias,self.X),axis=1)
	for j in range(Nb):
          grad=gradfunc(theta,j,bsize)
	  grad=grad.reshape(-1,1)
         #if LA.norm(grad)**0.5<tol: 
          #  print("Meeting tolerance criteria with tol:",tol,"breaking out at iteration=",i)
           # return theta
           # break
          theta += -grad*learning_rate
	  if self.costflag:self.costfunction(theta) 
      return theta
    
  def Normalfeature(self):
    print("~~~~~~Performing feature normalizing~~~~~~~~")
    nf=self.X.shape[1]
    self.mu=zeros(nf)
    self.std=zeros(nf)
    for i in range(nf):
      self.mu[i]=np.mean(self.X[:,i])
      self.std[i]=np.std(self.X[:,i])
      self.X[:,i]=(self.X[:,i]-self.mu[i])/self.std[i]
    #print(self.mu,self.std)
        
  def costfunction(self,param):
     linarg=np.dot(self.Xb,param)
     h=array([self.logisticfunc(z) for z in linarg])
     cost=1.0/self.nsample*( -np.dot(self.Y.transpose(),log(h) ) \
                            -np.dot(1.0-self.Y.transpose(),log(1.0-h)) )\
                           +self.C/(2*self.nsample)*(LA.norm(self.theta)-\
                           self.theta[0]**2) ##Adding regularized part to cost function
     
     if self.costflag:
   	i_cost= np.asscalar(cost) 
   	self.cost.append(i_cost)
     return np.asscalar(cost)
     
        
        
  def predict_prob(self,param,xn):
    """
    param is the trained theta
    xn should have the shape (ntest,nfeature)
    Return the prediction for new data xn
    """
    xn=xn.astype(float)  ##this line is necessary,which is somehow strange
    if self.nfflag:
        nf=xn.shape[1]
        for i in range(nf):
           xn[:,i]=(xn[:,i]-self.mu[i])/self.std[i]
    z= param[0]+np.dot(xn,param[1:]) 
    return self.logisticfunc(z) 
  
  def score(self,x,y,param):
    """
    param is the trained theta
    x should have the shape (ntest,nfeature)
    y should have the shape (ntest,1)
    Return the prediction for known data x and its label y
    """
    y=y.reshape(-1,1)   
    pred=self.predict_prob(param,x)>0.5
    return ( float(sum(pred==y))/len(y) )
    

  
  def logisticfunc(self,z):
    return 1.0/(1.0+exp(-z))
  
  def Decisionboundary(self,param):
      admits=self.X[(self.Y==1)[:,0] ]
      notadmits=self.X[(self.Y==0)[:,0] ]
      title("Normalized scale",fontsize='large')
      plot(admits[:,0],admits[:,1],'ro',label='Admit')
      plot(notadmits[:,0],notadmits[:,1],'bs',label='NotAdmit')
      legend(loc=0,fontsize='large')
      xlabel('score_1',size='large')
      ylabel('score_2',size='large')
      ##plot decision boundary
      x1min,x1max=min(self.X[:,0]),max(self.X[:,0])
      bdy_x=[x1min,(x1min+x1max)/2.0,x1max]
      bdy_y=[-(param[0]+param[1]*x)/param[2] for x in bdy_x]
      plot(bdy_x,bdy_y)
    #  print(x1min,x1max)

    
  

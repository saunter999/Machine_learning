#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import *
from pylab import *
import pandas as pd
import numpy as np
import time
from LogRegressor import LogRegress as LR

def preview(df):
  data=df.values
  admits=data[data[:,2]==1,:2]
  notadmits=data[data[:,2]==0,:2]
  plot(admits[:,0],admits[:,1],'ro',label='Admit')
  plot(notadmits[:,0],notadmits[:,1],'bs',label='NotAdmit')
  legend(loc=0,fontsize='large')
  xlabel('score_1',size='large')
  ylabel('score_2',size='large')
  
  
if __name__=="__main__":
    t1=time.time()

    url='https://raw.githubusercontent.com/saunter999/datasets/master/score_admit.csv'
    df=pd.read_csv(url,header=None,names=['score1','score2','admit_status'])

    figure(1)
    preview(df)
   
    X,Y,Nit,tol,alpha=df.values[:,:-1],df.values[:,-1],10,1e-6,0.2 
    nsample=Y.shape[0]
    mylogReg=LR(X,Y,C=0.01,batch_size=100)
    mthls=['SGD','BFGS','Newton-CG']
    mtd=mthls[0]
    print("~~~~~Training using "+mtd+"~~~~~")
    mylogReg.training(Nit,tol,alpha,method=mtd)
    
    
    figure(2)
    mylogReg.Decisionboundary(mylogReg.theta)
    
    figure(3)
    plot(range(len(mylogReg.cost)),mylogReg.cost,'co-',markersize=2)
    xlabel("Nit",size='large')
    ylabel("J_cost",size='large')
    print("Trained Theta=",mylogReg.theta)
    print ( "Prediction value for score(45,85)=",mylogReg.predict_prob( mylogReg.theta,array([[45,85]]) ) )
    #Accuracy for correct prediction on the whole training set
    score=mylogReg.score(X,Y,mylogReg.theta)
    print("Accuracy aka.correct prediction on the whole training set:",score)
    t2=time.time()
    print("Running time:",t2-t1)
    show()

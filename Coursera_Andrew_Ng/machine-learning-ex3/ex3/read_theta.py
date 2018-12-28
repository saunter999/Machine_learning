#!/usr/bin/env python
from scipy import *



if __name__=="__main__":
     Theta1=loadtxt('Theta1.txt') 
     Theta2=loadtxt('Theta2.txt') 
     Xinp=loadtxt('Xinp.txt') 
     Yout=loadtxt('Yout.txt') 
     print Theta1.shape
     print Theta2.shape
     print Xinp.shape
     print Yout.shape

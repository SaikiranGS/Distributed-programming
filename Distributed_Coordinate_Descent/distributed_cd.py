# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:14:52 2018

@author: saikiran
"""

from mpi4py import MPI
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

def rmse(x,y,theta):
    hypothesis = np.dot(x,theta)
    cost = np.average((y-hypothesis) ** 2)
    rmse = (math.sqrt(cost))
    return rmse

def safe_div(x,y):
    if y == 0:
        rem = 0
    else:
        rem = x/y
    return rem  
    
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.Get_size()
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
np.random.seed(0)
if rank == 0:
    data = pd.read_csv('C:\\Users\\saikiran\\Downloads\\cup98lrn\\cup98LRN.txt',sep=",",low_memory=False)
    char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index
    for c in char_cols:
        data[c] = pd.factorize(data[c])[0]
    
    #fills all NAN values with 0
    preprocessed_data=data.fillna(0)
    Y= preprocessed_data['TARGET_D']
    #X = preprocessed_data.drop(['ODATEDW', 'DOB','TARGET_B', 'TARGET_D','HPHONE_D'], axis=1)
    X = preprocessed_data.drop(['TARGET_D'], axis=1)
    X = np.array(X)
    #X = X[:40000]
    #print(X.shape)
    Y = np.array(Y)
    #Y = Y[:40000]
    #bias = np.random.rand(len(X),1)
    bias = np.ones((len(X),1), dtype=int)
    X = np.concatenate((bias,X),axis=1)
    Y = Y.reshape(len(Y),1)
    np.random.seed(0)
    #beta = np.random.rand(X.shape[1],1)
    beta = np.random.uniform(0.000005,0.00001,X.shape[1])
    beta = beta.reshape(len(beta),1)
    #beta = np.around(beta, decimals=3)
    a = int(len(X)/size)
    #23853
    X_chunks = [X[x:x+a] for x in range(0, len(X), a)]
    Y_chunks = [Y[x:x+a] for x in range(0, len(Y), a)]
    scat_variable_X = [(X_chunks[i])for i in range(len(X_chunks))]
    scat_variable_Y = [(Y_chunks[i])for i in range(len(Y_chunks))]
    bcast_variable_beta = beta
else:
    scat_variable_X = None
    scat_variable_Y = None
    bcast_variable_beta = None
receive_X = comm.scatter(scat_variable_X, root=0)
receive_Y = comm.scatter(scat_variable_Y, root=0)
theta = comm.bcast(bcast_variable_beta,root=0)
X_train, X_test, Y_train, Y_test = train_test_split(receive_X, receive_Y, test_size=0.30, random_state=42)
Rmse_train_list =[]
Rmse_test_list =[]
x_trans = X_train.T
max_epochs = 5
for j in range(max_epochs):
    for i in range(len(x_trans)):
        x = x_trans[i]
        x = x.T
        x=x.reshape(len(x),1)
        beta_parameter = theta[i]
        x_trans = np.delete(x_trans,i,0)
        theta = np.delete(theta,i,0)
        X_train = x_trans.T
        beta_parameter = x * (Y_train-np.dot(X_train,theta))
        beta_parameter = beta_parameter.sum(axis=0)
        x_sqr = (x**2)
        beta_parameter = safe_div(beta_parameter,(x_sqr.sum(axis=0)))
        theta = np.insert(theta,i,beta_parameter)
        theta = theta.reshape(len(theta),1)
        root=0
        local_beta = comm.reduce(theta,root=root,op=MPI.SUM)
        if rank==root:
            global_beta = local_beta/size  
        else:
            global_beta=None
        theta = comm.bcast(global_beta,root=root)
        x_trans = np.insert(x_trans,i,x.T,0)
        X_train = x_trans.T
    RMSE_train = rmse(X_train,Y_train,theta)
    RMSE_test = rmse(X_test,Y_test,theta)
    Rmse_train_list = np.append(Rmse_train_list,RMSE_train)
    Rmse_test_list = np.append(Rmse_test_list,RMSE_test)
    root=0
    train_rmse = comm.reduce(Rmse_train_list,root=root,op=MPI.SUM)
    test_rmse = comm.reduce(Rmse_test_list,root=root,op=MPI.SUM)
    end_time = MPI.Wtime()
    execution_time = end_time-start_time
    exe_epoch = comm.reduce(execution_time,root=root,op=MPI.MAX)
    if rank==root:
        print("epoch ={},total execution after each epoch is{}:".format(j,exe_epoch))
        global_train_Rmse = [x / size for x in train_rmse]
        for i in range (len(global_train_Rmse)):
            if i==0:
                print("epoch={} and total train_RMSE={}".format(j,global_train_Rmse[i]))
            else:
                rmse_diff =abs(global_train_Rmse[i]-global_train_Rmse[i-1])
                if rmse_diff<0.01:
                    print("training converged at epoch={}".format(i))
                    break
                else:
                    print("epoch={} and total train_RMSE={}".format(j,global_train_Rmse[i]))
            
        global_test_Rmse = [y / size for y in test_rmse]
        
print(global_train_Rmse)
print(global_test_Rmse)

   
    

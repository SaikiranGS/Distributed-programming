# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:15:37 2018

@author: saikiran
"""
from mpi4py import MPI
import numpy as np
def divide_data(data,n):
    split_data = np.split(data,n)
    return split_data
    
comm = MPI.COMM_WORLD
rank = comm.rank
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
n=4
if rank == 0:
    np.random.seed(0)
    vector1 = np.random.rand(4,4)
    vector1=np.ravel(vector1) 
    print("matrix1 is:",vector1)
    vector2 = np.random.rand(4,4)
    vector2=np.ravel(vector2)
    print("matrix2 is:",vector2)
    v1=divide_data(vector1,n)
    v2=divide_data(vector2,n)
    scat_variable1=[(v1[i])for i in range(n)]
    scat_variable2=[(v2[i])for i in range(n)]
else:
    scat_variable1 = None
    scat_variable2= None
recv1 = comm.scatter(scat_variable1, root=0)
size=len(recv1)
recv1 = np.reshape(recv1,(int(size/n),n))
recv2 = comm.scatter(scat_variable2, root=0)
recv2 = np.reshape(recv2,(int(size/n),n))
print("process={},variable shared from matrix1={}".format(rank,recv1))
print("process={},variable shared from matrix2={}".format(rank,recv2))
vector3= np.multiply(recv1,recv2)
print("product is :" , vector3)
root=0
data = comm.gather(vector3,root=root)
end_time = MPI.Wtime()
print("end time is:",end_time)
print("total execution time is :",end_time-start_time)



if rank ==root:
    final_data = np.reshape(data,(n,n))
    print(final_data)














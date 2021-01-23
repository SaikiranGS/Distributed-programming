# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:16:50 2018

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
np.random.seed(0)
vector1 = np.random.rand(1,16)
vector1=np.ravel(vector1)
vector2 = np.random.rand(1,16)
vector2=np.ravel(vector2)


for i in range(n):
    
    v1 = divide_data(vector1,n)
    v2 = divide_data(vector2,n)
    if rank==0:
        if i==0:
            data=np.add(v1[i],v2[i])
            print("my vector sum is:",data)
            end_time = MPI.Wtime()
            print("end time is:",end_time)
            print("total execution time is :",end_time-start_time)
        
        destination_process= i+1
        if destination_process==n:
            print("Data has been sent to all processes succesfully")
        else:
            comm.send(v1[i+1],dest=destination_process, tag=8)
            comm.send(v2[i+1],dest=destination_process, tag=9)
            print("sending vector1 data {} data to process{}" .format(v1[i+1],destination_process))
            print("sending vector2 data {} data to process{}" .format(v2[i+1],destination_process))
            final_vector=comm.recv(source = i+1,tag=4)
            print("received vector_sum data is",final_vector)
            append_data = np.append(data,final_vector)
            data = append_data
            print("my final_vector is :",data)
    if rank==i+1:
        vector3 = comm.recv(source=0,tag=8)
        print("received vector1 data is",vector3)
        vector4 = comm.recv(source=0,tag=9)
        print("received vector2 data is",vector4)
        data2 = np.add(vector3,vector4)
        destination_process = 0
        comm.send(data2, dest=destination_process,tag=4)
        print("sending vector sum data {} data to process{}" .format(data2,destination_process))
        print("my  vector sum is:", data2)
        end_time = MPI.Wtime()
        print("end time is:",end_time)
        print("total execution time is :",end_time-start_time)
        
if rank==n-1:
    print("vector sum using parallel processes is completed successfully")


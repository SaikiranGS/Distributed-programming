# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:49:25 2018

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
for i in range(n):
    v1 = divide_data(vector1,n)
    if rank==0:
        
        if i==0:
            data=np.average(v1[i])
            print("my vector average is:",data)
            end_time = MPI.Wtime()
            print("end time is:",end_time)
            print("total execution time is :",end_time-start_time)
        
        destination_process= i+1
        if destination_process==n:
            print("Data has been sent to all processes succesfully")
        else:
            comm.send(v1[i+1],dest=destination_process)
            print("sending vector1 data {} data to process{}" .format(v1[i+1],destination_process))
            final_vector=comm.recv(source = i+1)
            print("received vector_average data is",final_vector)
            append_data = np.append(data,final_vector)
            data = append_data
            print("my final_vector is :",data)
           
        
    if rank==i+1:
        vector3 = comm.recv(source=0)
        print("received vector1 data is",vector3)
        data2 = np.average(vector3)
        print("my  vector average is:", data2)
        destination_process = 0
        comm.send(data2, dest=destination_process)
        print("sending vector average data {} data to process{}" .format(data2,destination_process))
        end_time = MPI.Wtime()
        print("end time is:",end_time)
        print("total execution time is :",end_time-start_time)
        
if rank==n-1:
    print("vector average using parallel processes is completed successfully")



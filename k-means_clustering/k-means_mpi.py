# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:41:17 2018

@author: saikiran
"""


from mpi4py import MPI
import math
import json
import numpy as np
import os
from collections import Counter



def mean_array(path,mean):
    y = len(mean)
    cnt = [Counter() for i in range (y)]
    i=0
    for x in mean:
        filename=(path+x)
        mean_x = open(filename,'rt').read()
        mean_score = Counter(json.loads(mean_x))
        cnt[i] += mean_score
        if(i<y):
            i=i+1
    return cnt

    
def euclidean_distance(mean,data): 
    ed_list=[]
    for i in mean:
        c= math.sqrt(sum((i.get(d,0) - data.get(d,0))**2 for d in set(i) | set(data)))
        ed_list = np.append(ed_list,c)    
    return ed_list
        

def safe_div(x,y):
    if y == 0:
        rem = 0
    else:
        rem = x/y
    return rem  


def distortion_calculation(a):
    lm_list=[]
    for lm in a:
        lm = sum(lm.values())
        lm_sqr = (lm)**2
        lm_list =np.append(lm_list,lm_sqr)
    sum_list= lm_list.tolist()
    return sum(sum_list)
             
comm = MPI.COMM_WORLD
rank = comm.rank
size =comm.Get_size()
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
k=2
dist_list=[]
count= np.zeros(k)
count = count.tolist()

if rank == 0:
    dir_list = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf.idf')
    a = int(len(dir_list)/size)
    np.random.seed(0)
    mean = np.random.choice(dir_list, k, replace=False)
    mean_lst = mean_array('C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf.idf\\',mean)
    bcast_variable = mean_lst
    chunks = [dir_list[x:x+a] for x in range(0, len(dir_list), a)]
    scat_variable = [(chunks[i])for i in range(len(chunks))]
else:
    bcast_variable = None
    scat_variable = None
receive1 = comm.scatter(scat_variable, root=0)
receive2 = comm.bcast(bcast_variable,root=0)
global_mean=receive2
iteration = 0
while iteration<15:
    mean_update = [Counter() for n in range (k)]
    local_mean=[Counter() for n in range (k)]
    c1 =Counter()
    m=0
    n=0
    for each_file in receive1:
        filename = 'C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf.idf\\' +each_file
        tf_idf_result = open(filename, 'rt').read()
        tf_idf_score = Counter(json.loads(tf_idf_result))
        ed_distance = euclidean_distance(global_mean,tf_idf_score)
        min_pos = np.argmin(ed_distance)
        for i in range(k): 
            if(i==min_pos):
                count[i]+=1
                mean_update[i] += tf_idf_score
                
    distortion = distortion_calculation(mean_update)
    dist_list = np.append(dist_list,distortion)
    root=0
    distortion_count = comm.reduce(dist_list,root=root,op=MPI.SUM)
    end_time = MPI.Wtime()
    execution_time = end_time-start_time
    print("total execultion time is:",execution_time) 
    total_parallel_time = comm.reduce(execution_time,root=root,op=MPI.SUM)
    dist_diff=0
    if rank==root:
        data = distortion_count
        for i in range (len(data)):
            if i==0:
                print("iteration={} and total distortion={}".format(i,data[i]))
            else:
                dist_diff =data[i]-data[i-1]
                if dist_diff>0:
                    print("converged at iteration={}".format(i))
                    break
                else:
                    print("iteration={} and total distortion={}".format(i,data[i]))
        total_time = total_parallel_time
        print("total_parallel_execution time is=:",total_time)
    if dist_diff>0:
        print("converged at iteration={}".format(i))
        break
    for j in mean_update:
        for x in j:
            c1[x] = safe_div(j[x],count[m])
        local_mean[n]+=c1
        c1=Counter()
        if n<k:
            n=n+1
        if m<k:
            m=m+1
    iteration=iteration+1
    global_mean = local_mean



    
                

          

  

    

        
   
   

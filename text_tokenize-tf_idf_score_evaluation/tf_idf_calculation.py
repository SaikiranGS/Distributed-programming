# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:14:50 2018

@author: saikiran
"""
from mpi4py import MPI
import json
import os
from collections import Counter

comm = MPI.COMM_WORLD
rank = comm.rank
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
if rank == 0:   
    idf_result= open('C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\idf_results\\idf_result.txt','rt').read()
    idf =  json.loads(idf_result)
    #print(type(idf))
    bcast_variable = idf
    #print(type(idf))
    dir_list = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf_results')
    #print(dir_list)
    chunks = [dir_list[x:x+5] for x in range(0, len(dir_list), 5)]
    #print(chunks)
    scat_variable = [(chunks[i])for i in range(len(chunks))]
else:
    bcast_variable = None
    scat_variable = None
receive1 = comm.scatter(scat_variable, root=0)
receive2 = comm.bcast(bcast_variable,root=0)
#print("process={},document shared is={}".format(rank,receive1))
#print("process={},document shared is={}".format(rank,receive2))
for each_document in receive1:
    arr = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf_results\\' + each_document)
    for each_file in arr:
        filename = 'C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf_results\\' + each_document +'\\'+each_file
        tf_results = open(filename, 'rt').read()
        tf = json.loads(tf_results)
        tf_idf = Counter(dict((k, v * receive2[k]) for k, v in tf.items() if k in receive2))
        #print(tf_idf)
        newpath = r'C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf_idf_results\\' + each_document +"\\"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open(newpath+each_file, 'wt') as f:
            f.write(json.dumps(tf_idf))
            f.close()
        
   
   

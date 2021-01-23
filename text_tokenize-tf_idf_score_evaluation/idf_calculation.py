# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 23:38:19 2018

@author: saikiran
"""
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from mpi4py import MPI
import string
import os
from collections import Counter

comm = MPI.COMM_WORLD
rank = comm.rank
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
file_count = 0
#final_idf_data_list=[]
#final_idf_data_list = Counter(final_idf_data_list)
file_total ={}
file_total = Counter(file_total)
if rank == 0:
    dir_list = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups')
    #print(dir_list)
    chunks = [dir_list[x:x+5] for x in range(0, len(dir_list), 5)]
    #print(chunks)
    scat_variable = [(chunks[i])for i in range(len(chunks))]
else:
    scat_variable = None
receive = comm.scatter(scat_variable, root=0)
print("process={},document shared is={}".format(rank,receive))
for each_document in receive:   
    arr = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups\\' + each_document)
    for each_file in arr:
        
        filename = 'C:\\Users\\saikiran\\Downloads\\20_newsgroups\\' + each_document +'\\'+each_file
        file = open(filename, 'rt')
        text = file.read()
        file.close()
        #splitting to words
        tokens = word_tokenize(text)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        #print(words)
        end_time = MPI.Wtime()
        #print("end time is:",end_time)
        #print("total execution time is :",end_time-start_time)
        idf_score = Counter(words)
        file_count += 1
        #total_score = sum(tf_score.values())
        for x in idf_score:
            if idf_score[x]>0:
                idf_score[x] = 1
            else:
                idf_score[x]= 0
        file_total = file_total+idf_score
            
root=0
document_count = comm.reduce(file_count,root=root,op=MPI.SUM)
#print(document_count)
data = comm.reduce(file_total,root=root,op=MPI.SUM) 
if rank==root:
    total_number = document_count
    final_idf_data_list = data
    print("before",final_idf_data_list)
    for x in final_idf_data_list:
        final_idf_data_list[x] = math.log10((total_number)/final_idf_data_list[x])
    #print("after:",final_idf_data_list)
    newpath = r'C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\' +'\\idf_results\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open(newpath+"idf_result.txt","wt") as f:
        f.write(json.dumps(final_idf_data_list))
        f.close()
        
        
    


        
       
    
   
   

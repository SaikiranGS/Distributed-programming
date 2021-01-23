# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 23:38:19 2018

@author: saikiran
"""


from mpi4py import MPI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import string
import os
from collections import Counter


comm = MPI.COMM_WORLD
rank = comm.rank
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)



if rank == 0:
    dir_list = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups')
    chunks = [dir_list[x:x+5] for x in range(0, len(dir_list), 5)]
    print(chunks)
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
        tf_score = Counter(words)
        total_score = sum(tf_score.values())
        for x in tf_score:
            tf_score[x] = tf_score[x]/total_score
        #print(tf_score)
        
        newpath = r'C:\\Users\\saikiran\\Downloads\\20_newsgroups_test\\tf_results\\' + each_document +"\\"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open(newpath+each_file+".txt", 'wt') as f:
            f.write(json.dumps(tf_score))
            f.close()
        
   
   

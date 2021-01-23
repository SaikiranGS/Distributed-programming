# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:20:07 2018

@author: saikiran
"""

from mpi4py import MPI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
comm = MPI.COMM_WORLD
rank = comm.rank
print("my rank is:", rank)
start_time = MPI.Wtime()
print("start time is:",start_time)
n=20
if rank == 0:
    dir_list = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups')
    print(dir_list)
    scat_variable = [(dir_list[i])for i in range(n)]
else:
    scat_variable = None
receive = comm.scatter(scat_variable, root=0)
print("process={},document shared is={}".format(rank,receive))
arr = os.listdir('C:\\Users\\saikiran\\Downloads\\20_newsgroups\\' + receive)
for each_file in arr:
    filename = 'C:\\Users\\saikiran\\Downloads\\20_newsgroups\\' + receive +'\\'+each_file
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    #splitting the text to words
    tokens = word_tokenize(text)
    # convert to words to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    strip = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in strip if word.isalpha()]
    # filtering out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    print(words)
    end_time = MPI.Wtime()
    print("end time is:",end_time)
    print("total execution time is :",end_time-start_time)
    '''with open(filename,'w') as f:
        f.write(words)'''
   



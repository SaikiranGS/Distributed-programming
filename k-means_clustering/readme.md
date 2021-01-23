# Distributed Programming using Message Passing Interface (MPI)

The purpose of this project is to implement the distributed k-means cluster algorithm using MPI based on the tf-idf scores calculated on 20-news groupl article dataset. 

# Description about files
  1. k-means_mpi.py : Distributed K-means clustering algorithm using mpi on results obtained from tf_idf_calculation.py
  2. Distributed_k-means_clustering_mpi_report.pdf : This is the detailed description of k-means clustering implementation using mpi on tf-idf scores obtained using 20-news group article dataset
 
# MPI command to execute all programs

mpiexec -n "no:of workers(ex:1,2,4,..n)" python "FileName".py(extension)


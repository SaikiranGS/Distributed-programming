# Parallel Stochastic Gradient Descent using Message Passing Interface (MPI)

The purpose of this project is to implement parallel stochastic gradient descent (PSGD) algorithm algorithm using MPI on two different regression kaggle datasets. 

# Description about files
  1. linear_regression_on_kdd-cup_dataset.py : PSGD algorithm using mpi on kdd_cup_1998 dataset.
  2. linear_regression_on_virus_dataset.py  : PSGD algorithm using mpi on virus dataset.
  3. Parallel_SGD_using_mpi_implementation_report.pdf : PSGD implementaion  details on both the datasets using mpi are reported in this file.
  4. Parallel_SGD_Results_report.pdf : The experimental results obtained from both the datasets are reported here.
 
# MPI command to execute all programs

mpiexec -n "no:of workers(ex:1,2,4,..n)" python "FileName".py(extension)

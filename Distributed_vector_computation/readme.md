# Distributed Programming using Message Passing Interface (MPI)

The purpose of this project is to perform some basic vector and matrix computation on a distributed programming using MPI (point to point and collective communication). 
# Description about files
  1. vector_addition.py : Add two vectors and store results in a third vector using parallel processing MPI.
  2. vector_average.py: Find an average of numbers in a vector using parallel processing MPI.
  3. vector_multiplication.py: Parallel Operation using MPI-point to point communication.
  4. matrix_multiplication.py: Parallel Operation using MPI- collective communication
  
 
# MPI command to execute all programs

mpiexec -n "no:of workers(ex:1,2,4,..n)" python "FileName".py(extension)


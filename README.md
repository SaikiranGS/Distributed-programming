# Distributed Programming using Message Passing Interface (MPI)

The purpose of this project is to calculate the term frequency(TF), inverse document frequency(IDF), Term frequency-inverse document frequency (TF-IDF) scores of documents on a distributed programming using MPI. For this project implementation, I have used 20_news_group dataset.

# Description about files
  1. clean_tokenize.py : Data preprocesssing tasks like data cleaning and word tokenizing of large corpus using parallel processing MPI.
  2. tf_calc.py : Data preprocessing and calculating the term frequency score of each word inside the document using parallel processing MPI.
  3. idf_calc.py : Data preprocessing and calculating the inverse document frequency score score of each token present in all documents with respect to the corpus using Parallel processing MPI.
  4. tf_idf_calculation.py : Using the output of tf and idf, tf-idf score is calculated using parallel processing MPI.
  5. MPI_news_20_group.pdf : This is the detailed description of all the work I have done to implement the above described files.
 
# MPI command to execute the above programs

mpiexec -n "no:of workers(ex:1,2,4,..n)" python "FileName".py(extension)

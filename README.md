# New Cardinality Estimation Scheme

## Environment setup
  The following command using conda should setup the environment in linux CentOS.
  ```
  conda env create -f environment.yml
  ```
  If not, you need to manually download the following packages
  Required dependence: numpy, scipy, pandas, Pgmpy, pomegranate, networkx, tqdm, joblib, pytorch, psycopg2, scikit-learn, 
  Additional dependence: numba, bz2, Pyro (These packages are not required to reproduce the result in the paper.)
  
## Dataset download:
The optimal trained models for each dataset are already stored. If you are only interested in verifying the paper's result, you can skip the dataset download and model training, and directly execute the evaluate the learnt model.
1. STATS dataset:
   The STATS dataset can be found at: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/datasets/stats_simplified
   
   The STATS-CEB benchmark can be found at: 
   https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/workloads

2. IMDB dataset:
   The imdb dataset can be downloaded here: http://homepages.cwi.nl/~boncz/job/imdb.tgz
   
## Reproducing result on IMDB:
  We are currently working on the code cleaning for IMDB workload and merge it with STATS branch, so the code is a bit messy now.
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
  ### First run the following command to train the models
  ```
  python run_experiment.py --dataset imdb
         --generate_models
         --data_path /home/ubuntu/data_CE/imdb/{}.csv
         --model_path /home/ubuntu/data_CE/CE_scheme_models/
         
  ```
  data_path: the imdb dataset you just downloaded

  model_path: the location to save the model
  
  n_bins: number of bins to bucketize each key group. The default is None, where the predetermined number of bins will be used.

  ### Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset imdb
         --evaluate
         --model_path /home/ubuntu/data_CE/CE_scheme_models/model_imdb_default.pkl
         --query_file job_queries/all_queries.pkl
         --query_sub_plan_file job_queries/all_sub_plan_queries_str.pkl
         --SPERCENTAGE 1.0
         --query_sample_location job_queries/materialized_sample/{}/job/all_job/
         --save_folder /home/ubuntu/data_CE/CE_scheme_models/model_imdb_default_est.txt
  ```
  model_path: the location for the saved model
  
  query_file_location: the location of the sql queries,

  query_sub_plan_file: we provided the location of all sub-plan queries. If you would like to generate this file 
  yourself, please check the `get_job_sub_plan_queires` function in Evaluation.testing.py

  SPERCENTAGE: the sample rate for base-table cardinality estimation

  query_sample_location: the location of materialized sample. The default is None, where we will perform sample on the fly.
  
  save_folder: where to save the prediction

  
  ### End-to-end performance
  First, make sure you set up the docker environment for hacked Postgres: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
  
  Then run the following command to send the estimated results into docker container
  ```
  sudo docker cp /home/ubuntu/data_CE/CE_scheme_models/[method].txt ce-benchmark:/var/lib/pgsql/13.1/data/[method].txt
  ```
  
  `/home/ubuntu/data_CE/CE_scheme_models/[method].txt` is the location of the saved cardinality predictions

  Execute the follow command to get the end-to-end results:
  ```
  python send_query.py --dataset imdb
         --method_name [method].txt
         --query_file job_queries/all_queries_original.sql
         --save_folder job_queries/results/
  ```
  
  In order to reproduce the results, make sure to excute the query multiple time first to rule out the effect of the postgres cache and make fair comparisons among all methods.
  
# New Cardinality Estimation Scheme

## Environment setup
  The following command using conda should setup the environment in linux CentOS.
  ```
  conda env create -f environment.yml
  ```
  If not, you need to manually download the following packages
  Required dependence: numpy, scipy, pandas, Pgmpy, pomegranate, networkx, tqdm, joblib, pytorch, psycopg2, scikit-learn, numba 
  
  
## Dataset download:
We use two query workloads to evalute our results, STATS-CEB and IMDB-JOB.

1. STATS dataset:
   
   Clone the cardinality estimation benchmark repo:
   
   ```
   git clone https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
   ```
    
   The STATS dataset can be found at: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/datasets/stats_simplified
   
   After downloading the dataset, execute the following script to convert the date_time into integers.
   ```
   python run_experiment.py --dataset stats --preprocess_data 
                            --data_folder /home/ubuntu/End-to-End-CardEst-Benchmark/datasets/stats_simplified/
   ```
   
   The STATS-CEB benchmark query workload can be found at: 
   https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/workloads

2. IMDB dataset:
   
   To evaluate our model on IMDB dataset, please checkout branch imdb:
   
   ```
   git checkout imdb
   ```
    
   The imdb dataset can be downloaded here: http://homepages.cwi.nl/~boncz/job/imdb.tgz
   
## Reproducing result on STATS:

  In order to reproduce the result for STATS, 
  
  ### First run the following command to train the models
  ```
  python run_experiment.py --dataset stats
         --generate_models
         --data_path /home/ubuntu/End-to-End-CardEst-Benchmark/datasets/stats_simplified/{}.csv
         --model_path /home/ubuntu/data_CE/CE_scheme_models/
         --n_bins 200
         --bucket_method greedy
  ```
  data_path: the stats dataset you just downloaded

  model_path: the location to save the model
  
  n_bins: number of bins to bucketize each key group
  
  bucket_method: binning method, can choose between "greedy", "sub_optimal", and "naive". "greedy" is the binning algorithm explained in the paper. "sub_optimal" is a fast approaximation of "greedy" algorithm. "naive" is only used for ablation study, will not have good performance.

  
  ### Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset stats
         --evaluate
         --model_path /home/ubuntu/data_CE/CE_scheme_models/model_stats_greedy_200.pkl
         --query_file_location /home/ubuntu/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql
         --save_folder /home/ubuntu/data_CE/CE_scheme_models/
  ```
  model_path: the location for the saved model
  
  query_file_location: the sql file containing the queries
  
  save_folder: where to save the prediction
  
  ### End-to-end performance
  First, make sure you set up the docker environment for hacked Postgres: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
  
  Then run the following command to send the estimated results into docker container
  ```
  sudo docker cp /home/ubuntu/data_CE/CE_scheme_models/[method].txt ce-benchmark:/var/lib/pgsql/13.1/data/[method].txt
  ```
  Execute the follow command to get the end-to-end results:
  ```
  python send_query.py --dataset stats
         --method_name [method].txt
         --query_file /home/ubuntu/End-to-End-CardEst-Benchmark/workloads/stats_CEB/stats_CEB.sql
         --save_folder /home/ubuntu/data_CE/stats_CEB/
  ```
  
  In order to reproduce the results, make sure to excute the query multiple time first to rule out the effect the postgres cache and make fair comparisons among all methods.
  

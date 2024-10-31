# FactorJoin: A New Cardinality Estimation Framework for Join Queries


## Environment setup
  We use python version 3.7 and all required packages are specified in requirements.txt.

  ```
  pip install -r requirements.txt
  ```

  You can use anaconda (https://docs.anaconda.com/free/anaconda/install/index.html) to create a specifical python environment.
  ```
  conda create -n factorjoin python=3.7
  conda activate factorjoin
  ```

  For end-to-end evaluation, please set up the docker container for hacked Postgres: 
  https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
  
  
## Dataset download:
We use two query workloads to evalute our results, STATS-CEB and IMDB-JOB.

1. STATS dataset:
   
   Clone the cardinality estimation benchmark repo:
   
   ```bash
   git clone https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
   ```
    
   The STATS dataset can be found at: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/datasets/stats_simplified
   
   After downloading the dataset, execute the following script to convert the date_time into integers.
   ```bash
   python run_experiment.py --dataset stats --preprocess_data \
                            --data_folder /home/ubuntu/End-to-End-CardEst-Benchmark/datasets/stats_simplified/
   ```
   
   The STATS-CEB benchmark query workload can be found at: 
   https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master/workloads

2. IMDB dataset:
   The imdb dataset can be downloaded here: http://homepages.cwi.nl/~boncz/job/imdb.tgz
   The JOB query workload can be downloaded from: https://db.in.tum.de/~leis/qo/job.tgz

3. SSB dataset:
   The Star-Schema-Benchmark can be found here: https://github.com/electrum/ssb-dbgen
   ```
   git clone https://github.com/electrum/ssb-dbgen
   cd ssb-dbgen
   make
   ./dbgen -s 1 -T a
   ```
   You can choose the scale factor by setting -s to a different value, then convert the .tbl file to a .cvs for each 
   table t.
   ```
   sed 's/|$//' t.tbl > t.csv
   ```

   
## Reproducing result on STATS-CEB:

  In order to reproduce the result for STATS-CEB, 
  
  ### First run the following command to train the models
  ```
  python run_experiment.py --dataset stats \
         --generate_models \
         --data_path /home/ubuntu/End-to-End-CardEst-Benchmark/datasets/stats_simplified/{}.csv \
         --model_path checkpoints/ \
         --n_dim_dist 2 \
         --n_bins 200 \
         --bucket_method greedy
  ```
  data_path: the stats dataset you just downloaded

  model_path: the location to save the model

  n_dim_dist: the dimension of distributions (section 5.1 of the paper), i.e. the tree-width of the 
              Bayesian factorization. We currently only support 1 or 2.
  
  n_bins: number of bins to bucketize each key group
  
  bucket_method: binning method, can choose between "greedy", "sub_optimal", and "naive". "greedy" is the binning 
  algorithm explained in the paper. "sub_optimal" is a fast approaximation of "greedy" algorithm. "naive" is only used 
  for ablation study, will not have good performance.

  
  ### Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset stats \
         --evaluate \
         --model_path checkpoints/model_stats_greedy_200.pkl \
         --query_file_location /home/ubuntu/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql \
         --save_folder checkpoints/
  ```
  model_path: the location for the saved model
  
  query_file_location: the sql file containing the queries
  
  save_folder: where to save the prediction
  
  ### End-to-end performance
  First, make sure you set up the docker environment for hacked Postgres: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
  
  Then run the following command to send the estimated results into docker container
  ```
  sudo docker cp checkpoints/[method].txt ce-benchmark:/var/lib/pgsql/13.1/data/[method].txt
  ```
  
  /home/ubuntu/data_CE/CE_scheme_models/[method].txt is the location of the saved cardinality predictions
  
  Execute the follow command to get the end-to-end results:
  ```
  python send_query.py --dataset stats \
         --method_name [method].txt \
         --query_file /home/ubuntu/End-to-End-CardEst-Benchmark/workloads/stats_CEB/stats_CEB.sql \
         --save_folder checkpoints/
  ```
  
  In order to reproduce the results, make sure to execute the query multiple time first to warm up postgres and make fair comparisons among all methods.
  
  ### Model Update
  Run the following command to train a FactorJoin on data before 2014 and incrementally update the model with data after 2014:
  ```
  python run_experiment.py --dataset stats \
         --update_evaluate \
         --data_path /home/ubuntu/End-to-End-CardEst-Benchmark/datasets/stats_simplified \
         --model_path checkpoints/update/ \
         --n_dim_dist 2 \
         --n_bins 200 \
         --bucket_method sub_optimal \
         --split_date '2014-01-01 00:00:00'
  ```
  Afterwards, an updated model should be saved under --model_path, and you can follow the previous instruction to evaluate its end-to-end performance.
  

## Reproducing result on IMDB-JOB

As discussed in the paper, since IMDB-JOB contains complicated cyclic joins and complex predicates (disjunction, LIKE), 
most existing learned cardinality estimators can handle it. FactorJoin also needs to make certain qualifications to 
support it, including using sampling for base-table estimates.

### Get the subplan queries of IMDB-JOB
We provide the subplan queries in the checkpoints/derived_query_file.pkl, that you can directly load.

If you are interested in how they are generated, you can refer to 
https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark#how-to-generate-sub-plan-queries

### First run the following command to train the models
  ```
  python run_experiment.py --dataset imdb \
         --generate_models \
         --data_path /home/ubuntu/data_CE/imdb/{}.csv \
         --model_path checkpoints/ \
         --n_dim_dist 1 \
         --bucket_method fixed_start_key \
         --db_conn_kwargs "dbname=imdb user=postgres password=postgres host=127.0.0.1 port=5432"
  ```
  data_path: the stats dataset you just downloaded

  model_path: the location to save the model

  n_dim_dist: the dimension of distributions (section 5.1 of the paper), i.e. the tree-width of the 
              Bayesian factorization. We currently only support 1 for IMDB because it contains too many string columns
              and to the best of our knowledge there does not exist any work to capture the correlation between two string
              attributes (cannot discretize because of LIKE). We are exploring novel algorithm using n_dim_dist=2 in 
              optimization branch.
  
  bucket_method: binning method ["greedy", "fixed_start_key", "sub_optimal", and "naive"]. "fixed_start_key" is a fast
  approximation of GBSA and is recommended for IMDB-JOB workload because "greedy" is too slow.

  additional args:

  set --prepare_sample  for each training with different model parameters; this creates a new set of temporary tables in postgres

  set --materialize_sample and --query_file_location to pre-material a sample for the queries

### Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset imdb
         --evaluate
         --model_path checkpoints/model_imdb_default.pkl
         --derived_query_file checkpoints/derived_query_file.pkl
         --save_folder checkpoints/
         --query_sample_location checkpoints/binned_cards_{}/
  ```
  model_path: the location for the saved model
  
  query_file_location: the sql queries and their sub-plan queries
  
  save_folder: where to save the prediction

### End-to-end performance
  First, make sure you set up the docker environment for hacked Postgres: https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
  
  Then run the following command to send the estimated results into docker container
  ```
  sudo docker cp checkpoints/[method].txt ce-benchmark:/var/lib/pgsql/13.1/data/[method].txt
  ```
  
  /home/ubuntu/data_CE/CE_scheme_models/[method].txt is the location of the saved cardinality predictions
  
  Execute the follow command to get the end-to-end results:
  ```
  python send_query.py --dataset imdb
         --method_name [method].txt
         --query_file checkpoints/all_queries.sql
         --save_folder checkpoints/
  ```
  
  In order to reproduce the results, make sure to execute the query multiple time first to warm up postgres and make fair comparisons among all methods.


## Run IMDB-light schema and queries (e.g. JOB-light, JOBLightRanges)
We provided the IMDB-light schema in Schemas/imdb/schema.py. Check gen_job_light_imdb_schema function to see if the rows matches the IMDB dataset you downloaded.

### Preprocess IMDB dataset for JOBLightRanges
Since JOBLightRanges queries contain range filters on string columns, we need to convert the string to int type first.

Note that the current version of FactorJoin does not support range filters on string columns but it is not a fundamental
limitations and should be very easy to extend in the future.

```
python run_experiment.py --dataset imdb-light \
     --preprocess_data \
     --data_path ../../Data/imdb/{}.csv \
     --query_file_location checkpoints/JOBLightRangesQueries.sql \
     --save_folder ../../Data/imdb_tmp/
```

### Run the following command to train the models
```
python run_experiment.py --dataset imdb-light \
     --generate_models \
     --data_path ../../Data/imdb_tmp/{}.csv \
     --model_path checkpoints/ \
     --n_dim_dist 2 \
     --n_bins 200 \
     --bucket_method fixed_start_key \
     --get_bin_means
```
Note here the --data_path will be the path to your preprocessed IMDB-light dataset.
--bucket_method fixed_start_key is recommended for IMDB dataset due to training efficiency 


### Run the following command to test the models
```
python run_experiment.py --dataset imdb-light \
     --evaluate \
     --model_path checkpoints/model_imdb-light_fixed_start_key_200.pkl \
     --query_file_location ../../Data/imdb_tmp/JOBLightRangesQueries.sql
     --ground_true_file_location true_cards_joblightranges.csv
```

--query_file_location points to the rewritten JOBLightRangesQueries.sql



## Citation

This paper is accepted by SIGMOD 2023 (PACMMOD).

```
@inproceedings{factorjoin,
title = {{FactorJoin: A New Cardinality Estimation Framework for Join Queries}},
author={Ziniu Wu and Parimarjan Negi and  Mohammad Alizadeh and Tim Kraska and Samuel Madden},
journal={PACMMOD},
year = {2023},
}
```
    

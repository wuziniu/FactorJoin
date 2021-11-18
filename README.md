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
   
## Reproducing result on STATS:
  In order to reproduce the result for STATS, 
  First run the following command to train the models
  ```
  python run_experiment.py --dataset stats
         --generate_models
         --data_path /Users/ziniuw/Desktop/past_research/End-to-End-CardEst-Benchmark/datasets/stats_simplified/{}.csv
         --model_path /Users/ziniuw/Desktop/research/Learned_QO/CC_model/CE_scheme_models/stats/
         --n_bins 200
  ```
  data_path points the dataset you just downloaded

  model_path specifies the location to save the model

  
  Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset stats
         --evaluate
         --model_path /Users/ziniuw/Desktop/research/Learned_QO/CC_model/CE_scheme_models/stats/
         --query_file_location /Users/ziniuw/Desktop/past_research/End-to-End-CardEst-Benchmark/workloads/stats_CEB/stats_CEB.sql
  ```
  
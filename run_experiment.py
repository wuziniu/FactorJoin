import argparse
import logging
import os
import time
import shutil
import numpy as np
import pandas as pd

from Evaluation.training import train_one_imdb
from Evaluation.testing import test_on_imdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb', help='Which dataset to be used')

    # generate models/ensembles
    parser.add_argument('--generate_models', help='Trains BNs on dataset', action='store_true')
    parser.add_argument('--data_path', default='../data_CE/imdb/{}.csv')
    parser.add_argument('--model_path', default='../CE_scheme_models')
    parser.add_argument('--n_bins', type=int, default=None, help="The bin size on the id attributes")
    parser.add_argument('--save_bucket_bins', help="Whether want to support data update", action='store_true')

    # evaluation
    parser.add_argument('--evaluate', help='Evaluates models to compute cardinality bound', action='store_true')
    parser.add_argument('--model_location', nargs='+', default='../CE_scheme_models')
    parser.add_argument('--query_file', default='../job_queries/all_queries.pkl')
    parser.add_argument('--query_sub_plan_file', default='../job_queries/all_sub_plan_queries_str.pkl')
    parser.add_argument('--SPERCENTAGE', type=float, default=1.0)
    parser.add_argument('--query_sample_location', type=str, default=None)
    parser.add_argument('--save_folder', default='../job_queries/estimates.txt')

    # log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    print(args.dataset)
    if args.dataset == 'imdb':
        if args.generate_models:
            start_time = time.time()
            bound_ensemble = train_one_imdb(args.data_path, args.model_path, args.n_bins, args.save_bucket_bins)
            end_time = time.time()
            print(f"Training completed: total training time is {end_time - start_time}")

        elif args.evaluate:
            test_on_imdb(args.model_location, args.query_file, args.query_sub_plan_file, args.SPERCENTAGE,
                         args.query_sample_location, args.save_folder)




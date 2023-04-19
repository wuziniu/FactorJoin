# factorjoin-binned-cards

Following commands to run after setting up postgresql on appropriate port.

```bash
python3 scripts/create_binned_cols.py --port 5432 --pwd password --user ceb --sampling_percentage 1.0
python3 scripts/get_query_binned_cards.py --query_dir queries/job/all_job --port 5432 --pwd password --user ceb --sampling_percentage 1.0

```

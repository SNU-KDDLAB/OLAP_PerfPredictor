# OLAP_PerfPredictor
Train a model which predicts a performance of workload under a new configuration. 

Currently, it is tested with MySQL 8.0 database.
We collect train and test datasets as follows.

1. Changing configruations randomly
2. Reboot MySQL engine
3. Execute TPC-H 1GB 22 queries (Measure execution times of individual query and total workload)

Usage 
```
python mysql_perf_predict.py
```

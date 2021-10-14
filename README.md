# OLAP_PerfPredictor
Train a model which predicts a performance of workload under a new configuration. 
Our proposed prediction model utilizes an extra regularization method to predict the total execution time more precisely.
The paper "데이터베이스  시스템에서의 워크로드  수행  시간  예측  모델" is presented in KCC 2021.

Currently, it is tested with MySQL 8.0 database.


We collect train and test datasets as follows.

1. Changing configruations randomly
2. Reboot MySQL engine
3. Execute TPC-H 1GB 22 queries (Measure execution times of individual query and total workload)

Usage 
```
python mysql_perf_predict.py
```

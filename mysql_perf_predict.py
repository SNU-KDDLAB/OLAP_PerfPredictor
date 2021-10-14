# Author: Hanjun Goo, goohanjun@gmail.com
# Date  : 2021.10.14

import argparse
import pickle
import numpy as np
from glob import glob
from pandas import DataFrame
from pprint import pprint
import sys
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_pkl_dir",
        type=str,
        default='./train',
        help="directory for train pkls"
    )
    parser.add_argument(
        "--test_pkl_dir",
        type=str,
        default='./test',
        help="directory for test pkls"
    )
    return vars(parser.parse_args())

def read_pkl_from_dir(dir_path):
    pkls = glob(f'{dir_path}/*.pkl')
    data = []
    configs = []
    for pkl in pkls:
        with open(pkl, 'rb') as f:
            a = pickle.load(f)

        status_set = set()
        for q_id, q_result in a['raw_times'].items():
            status_set.add(q_result['status'])
        if len(status_set) != 1 or len(a['raw_times']) < 22:
            continue

        # featurize
        config = a['config']
        configs.append(config)
        flush_method = config.pop('innodb_flush_method', None)
        innodb_flush_methods = ["fsync", "O_DSYNC", "littlesync", "nosync", "O_DIRECT", "O_DIRECT_NO_FSYNC",]
        method_index = innodb_flush_methods.index(flush_method)

        x = np.zeros(len(config) + len(innodb_flush_methods) )

        for i, k in enumerate(sorted(config.keys())):
            x[i] = config[k]

        # one-hot encoding
        x[len(config) + method_index] = 1.

        y = np.zeros(23)  # 22 single query result + total benchmark
        for q_result in a['raw_times'].values():
            query_idx = q_result['q_type']  # 1-22
            y[query_idx] = q_result['query_run_time']

        y[0] = a['run_time']

        data.append((x, y))
    return data, configs


def convert_list2pair(data):
    Xs, Ys = np.zeros((len(data), len(data[0][0])) ), np.zeros((len(data), len(data[0][1])))
    for i, (x, y) in enumerate(data):
        Xs[i, :] = x
        Ys[i, :] = y
    data = (Xs, Ys)
    return data


def fit_nn_multi_model(train, valid, test, alpha=0.5):
    # use all query results
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test

    sc, sc_target = StandardScaler(), StandardScaler()

    train_x, train_y = sc.fit_transform(train_x), sc_target.fit_transform(train_y)
    valid_x, test_x = sc.transform(valid_x), sc.transform(test_x)

    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
    valid_x, test_x = torch.Tensor(valid_x), torch.Tensor(test_x)
    valid_x, test_x = Variable(valid_x), Variable(test_x)

    target_mean, target_scale = torch.Tensor([sc_target.mean_]), torch.Tensor([sc_target.scale_])

    n_samples, n_features = test_x.shape
    n_samples, n_outputs = test_y.shape

    net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, n_outputs),
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 64
    EPOCH = 500

    torch_dataset = Data.TensorDataset(train_x, train_y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2,)

    # start training
    best_result = {'test_y': test_y}

    min_test_error, min_valid_error = 1e8, 1e8
    patience = 0
    for epoch in range(EPOCH):
        train_loss = 0.
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x, b_y = Variable(batch_x), Variable(batch_y)

            normalized_prediction = net(b_x)     # input x and predict based on x
            loss = loss_func(normalized_prediction, b_y)     # must be (1. nn output, 2. target)

            prediction = (target_scale * normalized_prediction) + target_mean
            loss_aug = loss_func(prediction[:, 0], torch.sum(prediction[:, 1:], dim=1))
            tmp_loss = loss + alpha * loss_aug

            train_loss += tmp_loss
            train_error = torch.mean((prediction - b_y)**2.)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            print(f"train_error: {train_error:.2f} @ step {step} / epoch {epoch}", end='\r')

        predicted_y = net(valid_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        valid_error = np.mean((predicted_y[:, 0] - valid_y[:, 0])**2.)

        predicted_y = net(test_x)
        predicted_y = sc_target.inverse_transform(predicted_y.detach())
        test_error = np.mean((predicted_y[:, 0] - test_y[:, 0])**2.)
        print(f"valid error: {valid_error:.2f} / test error: {test_error:.2f} / epoch {epoch}")

        if min_valid_error > valid_error:
            best_result['prediction'] = predicted_y
            best_result['test_error'] = test_error
            min_valid_error = valid_error
            min_test_error = test_error
            patience = 0
        else:
            patience += 1
        if patience == 10:
            break
    return best_result


if __name__ == '__main__':

    args = load_args()

    train_data_list, train_configs = read_pkl_from_dir(args['train_pkl_dir'])
    test_data_list, test_configs = read_pkl_from_dir(args['test_pkl_dir'])
    print(f"Train: {len(train_data_list)}, Test: {len(test_data_list)} points are collected")

    if len(train_data_list) < 5:
        print("Train data is not enough")
        sys.exit()

    if len(test_data_list) == 0:
        print("Test data is not given")
        sys.exit()

    train_data = convert_list2pair(train_data_list)
    train_x, train_y = train_data
    train_size = int(train_x.shape[0] *0.8)

    train_data = (train_x[:train_size, :], train_y[:train_size, :])
    valid_data = (train_x[train_size:, :], train_y[train_size:, :])
    test_data = convert_list2pair(test_data_list)

    res = fit_nn_multi_model(train_data, valid_data, test_data, alpha=1.)

    for i, (test_config, pred_y, true_y) in enumerate(zip(test_configs, res['prediction'][:, 0], res['test_y'][:,0])):
        if i == 5:
            break
        print(f"\n{i}-th sample in test set")
        print(f"{test_config}\n->> Predicted Total Exec Time: {pred_y:.2f} / True Exec Time: {true_y}")


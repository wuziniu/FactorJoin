import os
import errno
import torch
from torch.autograd import Variable
import copy
import numpy as np
import pandas as pd
import glob
import string
import hashlib
import pickle
import gzip
import re
import pdb
import random
# from utils.tf_summaries import TensorboardSummaries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def shuffle_list(*ls):
  l = list(zip(*ls))
  random.shuffle(l)
  return zip(*l)

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def is_float(val):
    try:
        float(val)
        return True
    except:
        return False

def extract_ints_from_string(string):
    return re.findall(r'\d+', string)

# Generalized from:
#https://stackoverflow.com/questions/18683821/generating-random-correlated-x-and-y-points-using-numpy
def gen_gaussian_data(means, covs, num):
    vals = np.random.multivariate_normal(means, covs, num).T
    for i, v in enumerate(vals):
        vals[i] = [int(x) for x in v]
    return list(zip(*vals))

def save_object(file_name, data, use_csv=False):
    if isinstance(data, pd.DataFrame) and use_csv:
        data.to_csv(file_name.replace(".pkl", ".csv"), sep="|", index=False,
                encoding="utf-8")
        return

    tmp_fn = file_name + ".tmp"
    with open(tmp_fn, "wb") as f:
        pickle.dump(data, f,
                protocol=4)
    os.rename(tmp_fn, file_name)

def save_object_gzip(file_name, data):
    # with open(file_name, "wb") as f:
        # res = f.write(pickle.dumps(data))
    tmp_fn = file_name + ".tmp"
    pickle.dump(data, gzip.open(tmp_fn, 'wb'))
    os.rename(tmp_fn, file_name)

def load_object_gzip(file_name):
    res = None
    if os.path.exists(file_name):
        f = gzip.GzipFile(file_name, 'rb')
        res = pickle.load(f)

    return res

def load_object(file_name):
    res = None
    if ".csv" in file_name:
        res = pd.read_csv(file_name, sep="|", encoding='utf-8')
    else:
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                res = pickle.loads(f.read())
    return res

def update_list(fn, new_obj):
    obj = load_object(fn)
    if obj is None:
        obj = []
    obj.append(new_obj)
    save_object(fn, obj)

def save_or_update(obj_name, obj):
    # TODO: generalize this functionality
    # dir_name = os.path.dirname(obj_name)
    # if not os.path.exists(dir_name):
        # make_dir(dir_name)
    saved_obj = load_object(obj_name)
    if saved_obj is None:
        saved_obj = obj
    else:
        if isinstance(saved_obj, dict):
            saved_obj.update(obj)
        elif isinstance(saved_obj, list):
            saved_obj.append(obj)
        else:
            # TODO: not sure best way to handle pandas
            saved_obj = saved_obj.append(obj)
    save_object(obj_name, saved_obj)

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def cosine_similarity_vec(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*
            np.linalg.norm(vec2))
    return cosine_similarity

def get_substr_words(words, substr):
    vals = []
    for w in words:
        if substr in w:
            vals.append(w)
    return vals

def get_regex_match_words(words, regex):
    vals = []
    for w in words:
        if regex.search(w) is not None:
            vals.append(w)
    return vals

def clear_terminal_output():
    os.system('clear')

def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        # arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad).to(device)
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    else:
        arr = Variable(arr, requires_grad=requires_grad)

    # if torch.cuda.is_available() and use_cuda:
        # print("returning cuda array!")
        # arr = arr.cuda()
    # else:
        # pdb.set_trace()
    return arr


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def copy_network(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2

def save_network(model, name, step, out_dir, delete_old=False):
    '''
    saves the model for the given step, and deletes models for older
    steps.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    make_dir(out_dir)
    # find files in the directory that match same format:
    fnames = glob.glob(out_dir + name + "*")
    if delete_old:
        for f in fnames:
            # delete old ones
            os.remove(f)

    # Save model
    torch.save(model.state_dict(), '{}/{}_step_{}'.format(out_dir, name, step))

def model_name_to_step(name):
    return int(name.split("_")[-1])

def get_model_names(name, out_dir):
    '''
    returns sorted list of the saved model_step files.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    # find files in the directory that match same format:
    fnames = sorted(glob.glob(out_dir + name + "*"), key=model_name_to_step)
    return fnames

def get_model_name(args):
    if args.suffix == "":
        return str(hash(str(args)))
    else:
        return args.suffix

def adjust_learning_rate(args, optimizer, epoch):
    """
    FIXME: think about what makes sense for us?
    Sets the learning rate to the initial LR decayed by half every 30 epochs
    """
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.5 ** (epoch // 30))
    lr = max(lr, args.min_lr)
    if (epoch % 30 == 0):
        print("new lr is: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_nn(net, X, Y, lr=0.00001, max_iter=10000, mb_size=32,
        loss_func=None, tfboard_dir=None, adaptive_lr=False,
        min_lr=1e-17, loss_threshold=1.0, test_size=1024, eval_iter=100):
    '''
    very simple implementation of training loop for NN.
    '''
    if loss_func is None:
        assert False
        loss_func = torch.nn.MSELoss()

    # if tfboard_dir:
        # make_dir(tfboard_dir)
        # tfboard = TensorboardSummaries(tfboard_dir + "/tflogs/" +
            # time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        # tfboard.add_variables([
            # 'train-loss', 'lr', 'mse-loss'], 'training_set_loss')

        # tfboard.init()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # update learning rate
    if adaptive_lr:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20,
                        verbose=True, factor=0.1, eps=min_lr)
        plateau_min_lr = 0

    num_iter = 0
    X = np.array(X)
    Y = np.array(Y)

    while True:
        if (num_iter % eval_iter == 0):
            # test on the full train set
            # xbatch = X
            idxs = np.random.choice(list(range(len(X))), test_size)
            xbatch = X[idxs]
            xbatch = to_variable(xbatch).float()
            ybatch = Y[idxs]
            ybatch = to_variable(ybatch).float()
            # xbatch = to_variable(xbatch).float()
            # ybatch = Y
            # ybatch = to_variable(ybatch).float()
            pred = net(xbatch)
            pred = pred.squeeze(1)
            train_loss = loss_func(pred, ybatch).item()
            mse_loss = torch.nn.functional.mse_loss(pred, ybatch).item()
            print("num iter: {}, num samples: {}, mse loss: {}, loss func: {}".format(
                num_iter, len(X), mse_loss, train_loss))

            cur_lr = optimizer.param_groups[0]['lr']
            if adaptive_lr:
                # FIXME: should we do this for minibatch / or for train loss?
                scheduler.step(train_loss)
                if cur_lr*0.1 <= min_lr:
                    plateau_min_lr += 1

            if train_loss < loss_threshold:
                print("breaking because train_loss < {}".format(loss_threshold))
                break
            if plateau_min_lr >= 5:
                print("breaking because min lr and learning stopped")
                break

            if tfboard_dir:
                tfboard.report(num_iter,
                    [train_loss, cur_lr, mse_loss], 'training_set_loss')

        idxs = np.random.choice(list(range(len(X))), mb_size)
        xbatch = X[idxs]
        xbatch = to_variable(xbatch).float()
        ybatch = Y[idxs]
        ybatch = to_variable(ybatch).float()

        pred = net(xbatch)
        pred = pred.squeeze(1)
        loss = loss_func(pred, ybatch)

        if (num_iter > max_iter):
            print("breaking because max iter done")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_iter += 1

    print("done with training")
    print("training loss: ", train_loss)



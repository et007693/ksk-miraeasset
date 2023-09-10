import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, user_train_side, usernum, itemnum,item_side, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq_side = np.zeros([maxlen], dtype=np.int32)
        
        pos = np.zeros([maxlen], dtype=np.int32)
        pos_side = np.zeros([maxlen], dtype=np.int32)

        neg = np.zeros([maxlen], dtype=np.int32)
        neg_side = np.zeros([maxlen], dtype=np.int32)
        
        nxt = user_train[user][-1]
        nxt_side = user_train_side[user][-1]

        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            seq_side[idx] = item_side.loc[i]['side']

            pos[idx] = nxt
            pos_side[idx] = nxt_side
            
            if nxt != 0:
              neg[idx] = random_neq(1, itemnum + 1, ts)
              neg_side[idx] = item_side.loc[neg[idx]]['side']

            nxt = i
            nxt_side = item_side.loc[i]['side']

            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg
        , seq_side, pos_side, neg_side)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, user_train_side, usernum, itemnum, item_side, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      user_train_side,
                                                      usernum,
                                                      itemnum,
                                                      item_side,                                                      
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      32
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(fname):
    usernum = 0
    itemnum = 0
    user_train = {}
    user_train_side = {}

    user_valid = {}
    user_valid_side = {}
    
    user_test = {}
    user_test_side = {}

    f = pd.read_csv('data/%s.csv'% fname, encoding = 'UTF-8')
    f = f[['index', 'Code', 'rate']]
    f.rename(columns = {'rate':'side'}, inplace = True)

    umap = {u: (i+1) for i, u in enumerate(set(f['index']))}
    smap = {s: (i+1) for i, s in enumerate(set(f['Code']))}

    f['index'] = f['index'].map(umap)
    f['Code'] = f['Code'].map(smap)

    item_side = f.drop(['index'],axis=1).drop_duplicates(['Code'])
    item_side = item_side.set_index('Code')

    User= f.groupby('index')['Code'].apply(list).reset_index()
    User_side= f.groupby('index')['side'].apply(list).reset_index()

    User=User.set_index('index').to_dict()['Code']
    User_side=User_side.set_index('index').to_dict()['side']

    usernum=len(umap)
    itemnum=len(smap)

    # 길이가 3 개 미만인 고객은 test, valid 제외
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train_side[user] = User_side[user]

            user_valid[user] = []
            user_valid_side[user] = []
            
            user_test[user] = []
            user_test_side[user] = []
            
        else:
            user_train[user] = User[user][:-2]
            user_train_side[user] = User_side[user][:-2]

            user_valid[user] = []
            user_valid_side[user] = []

            user_valid[user].append(User[user][-2])
            user_valid_side[user].append(User_side[user][-2])

            user_test[user] = []
            user_test_side[user] = []

            user_test[user].append(User[user][-1])
            user_test_side[user].append(User_side[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum,
    user_train_side, user_valid_side, user_test_side, item_side,
    umap, smap]

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum,
    train_side, valid_side, test_side, item_side,
    umap, smap] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_side = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        idx = args.maxlen - 1
        seq_side[idx] = valid_side[u][0]
        idx -= 1
        for i in reversed(train_side[u]):
            seq_side[idx] = i
            idx -= 1
            if idx == -1: break
        idx = args.maxlen - 1

        rated = set(train[u])
        rated.add(0)
        rated_side = set(train_side[u])
        rated_side.add(0)

        item_idx = [test[u][0]]
        item_side_idx = [test_side[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_side_idx.append(item_side.loc[t]['side'])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [seq_side], item_side_idx]])
        predictions = predictions[0] 

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum,
    train_side, valid_side, test_side, item_side,
    umap, smap] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_side = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        for i in reversed(train_side[u]):
            seq_side[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        rated_side = set(train_side[u])
        rated_side.add(0)
        
        item_idx = [valid[u][0]]
        item_side_idx = [valid_side[u][0]]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_side_idx.append(item_side.loc[t]['side'])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [seq_side], item_side_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 500 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def recommend_items(model, dataset, args):
    [train, valid, test, usernum, itemnum,
    train_side, valid_side, test_side, item_side,
    umap, smap] = copy.deepcopy(dataset)

    # train, valid, test 합치기 
    for key, value in valid.items():
        if key in train:
            train[key].extend(value)
        else:
            train[key] = value
    
    for key, value in test.items():
        if key in train:
            train[key].extend(value)
        else:
            train[key] = value
    user = range(1, usernum + 1)

    recommended_items1 = []
    recommended_items2 = []
    recommended_items3 = []
    recommended_items4 = []
    recommended_items5 = []

    for u in tqdm(user):
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_side = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        
        item_idx = valid[u]
        item_side_idx = valid_side[u]

        for t in range(1, itemnum + 1):
            if t not in rated:
                item_idx.append(t)
                item_side_idx.append(item_side.loc[t]['side'])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [seq_side], item_side_idx]])
        predictions = predictions[0]

        rank1 = predictions.argsort().argsort()[0].item()
        rank2 = predictions.argsort().argsort()[1].item()
        rank3 = predictions.argsort().argsort()[2].item()
        rank4 = predictions.argsort().argsort()[3].item()
        rank5 = predictions.argsort().argsort()[4].item()

        recommended_items1.append(rank1)
        recommended_items2.append(rank2)
        recommended_items3.append(rank3)
        recommended_items4.append(rank4)
        recommended_items5.append(rank5)
    
    recommended_data = pd.DataFrame({'item1' : recommended_items1, 'item2' : recommended_items2, 'item3' : recommended_items3, 'item4' : recommended_items4, 'item5' : recommended_items5}).reset_index()

    recommended_data['index'] = recommended_data['index'].map(umap)
    recommended_data['item1'] = recommended_data['item1'].map(smap)
    recommended_data['item2'] = recommended_data['item2'].map(smap)
    recommended_data['item3'] = recommended_data['item3'].map(smap)
    recommended_data['item4'] = recommended_data['item4'].map(smap)
    recommended_data['item5'] = recommended_data['item5'].map(smap)

    return recommended_data
import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing
from torch import Tensor
import torch
import pickle
from collections import defaultdict
import math


def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)


def view_back(M):
    return view3(M) if M.dim() == 2 else view2(M)


def cosine_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    time = set([0])
    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [int(item) for item in para]
            entity.add(head);
            entity.add(tail);
            rel.add(r + 1)
            time.add(ts + 1);
            time.add(te + 1)
            triples.append((head, r + 1, tail, ts + 1, te + 1))
        else:
            head, r, tail, t = [int(item) for item in para]
            entity.add(head);
            entity.add(tail);
            rel.add(r + 1)
            time.add(t + 1)
            triples.append((head, r + 1, tail, t + 1))
    return entity, rel, triples, time


def load_all_triples(file_path, reverse=True):
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i, 0] = triples[i, 2]
            reversed_triples[i, 2] = triples[i, 0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples

    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()

    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()

    triples = np.array([line.replace("\n", "").split("\t")[0:3] for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:, 0]), np.max(triples[:, 2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1

    all_triples = np.concatenate([triples, reverse_triples(triples)], axis=0)
    all_triples = np.unique(all_triples, axis=0)

    return all_triples, node_size, rel_size * 2 if reverse else rel_size


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def get_matrix(triples, entity, rel, time):
    ent_size = max(entity) + 1
    rel_size = (max(rel) + 1)
    time_size = (max(time) + 1)
    print(ent_size, rel_size, time_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    time_link = np.zeros((ent_size, time_size))  # new adding

    for i in range(max(entity) + 1):
        adj_features[i, i] = 1

    # 先进行判断，说明数据集中要么都是时间点，要么都是区间，后续可能需要改
    if len(triples[0]) < 5:
        for h, r, t, tau in triples:
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1;
            adj_features[t, h] = 1
            radj.append([h, t, r, tau]);
            radj.append([t, h, r + rel_size, tau])
            time_link[h][tau] += 1;
            time_link[t][tau] += 1
            rel_out[h][r] += 1;
            rel_in[t][r] += 1
    else:
        for h, r, t, ts, te in triples:
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1;
            adj_features[t, h] = 1
            radj.append([h, t, r, ts]);
            radj.append([h, t, r + rel_size, te])
            time_link[h][te] += 1;
            time_link[h][ts] += 1
            time_link[t][ts] += 1;
            time_link[t][te] += 1
            rel_out[h][r] += 1;
            rel_in[t][r] += 1
    count = -1
    s = set()
    d = {}
    r_index, t_index, r_val = [], [], []
    for h, t, r, tau in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    time_features = time_link
    time_features = normalize_adj(sp.lil_matrix(time_features))

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features


def load_data(lang, train_ratio=0.3, unsup=False):
    entity1, rel1, triples1, time1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2, time2 = load_triples(lang + 'triples_2')
    # modified here #

    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    if train_ratio < 0.25:
        train_ratio = int(len(train_pair) * train_ratio)
        dev_pair = train_pair[train_ratio:] + dev_pair
        train_pair = train_pair[:train_ratio]
        print(len(train_pair))
    if unsup:
        dev_pair = train_pair + dev_pair
        train_pair = load_alignment_pair(lang + 'unsup_link')

    adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_feature = \
        get_matrix(triples1 + triples2, entity1.union(entity2), rel1.union(rel2), time1.union(time2))

    return np.array(train_pair), np.array(dev_pair), adj_matrix, np.array(r_index), np.array(r_val), \
           np.array(t_index), adj_features, rel_features, time_feature

def saveobj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def cal_sims(test_pair, feature, right=None):
    if right is None:
        feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
        feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
    else:
        feature_a = feature[:right]
        feature_b = feature[right:]
    fb = tf.transpose(feature_b, [1, 0])
    return tf.matmul(feature_a, fb)

def getLast(filename, pos=0):
    f = open(filename, 'r', encoding='utf-8')
    last = f.readlines()[-1]
    last = int(last.split()[pos])
    return last + 1

def construct_sparse_rel_matrix(all_triples, node_size):
    dr = {}
    for x, r, y in all_triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1
    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])
    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])
    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2],
                                        dense_shape=(node_size, node_size))
    return sparse_rel_matrix

def get_time(filename):
    time_dict = dict()
    count = 0
    for i in [0, 1]:
        for line in open(filename + 'triples_' + str(i + 1), 'r'):
            words = line.split()
            head, r, tail, t1, t2 = [int(item) for item in words]
            t = t1 * 1000 + t2
            if t not in time_dict.keys():
                time_dict[t] = count
                count += 1
    return time_dict

def get_feature_matrix(filename, i: int, shift_id=0, tf_idf=False, time_id=None):
    count = 0
    time_dict = dict()
    TS = 1000000
    num_triple = 0
    time_set = set()
    entity_set = set()
    th = defaultdict(lambda: defaultdict(int))
    t_e = defaultdict(set)
    e_t = defaultdict(int)
    for line in open(filename + 'triples_' + str(i), 'r'):
        num_triple += 1
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
            t_encode = 1
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]
            t_encode = t1 * 1000 + t2
            t = time_id[t_encode]
        time_set.add(t)
        head, tail = head - shift_id, tail - shift_id
        entity_set.add(head)
        entity_set.add(tail)
        if t_encode > 0:
            th[head][t] += 1
            th[tail][t + TS] += 1
            t_e[t].add(head)
            t_e[t + TS].add(tail)
            e_t[head] += 1
            e_t[tail] += 1

    index, value = [], []
    # num_ent = len(entity_set)
    num_ent = getLast(filename + 'ent_ids_' + str(i))
    if i == 2:
        num_ent -= shift_id

    if time_id is not None:
        num_time = len(time_id.keys())
    else:
        num_time = len(time_set)
    for ent, dic in th.items():
        for time, cnt in dic.items():
            t = time if time < TS else time + num_time - TS  # different id for head and tail
            index.append((ent, t))
            if tf_idf:
                tf = cnt / e_t[ent]
                idf = math.log(num_ent / (len(t_e[time]) + 1))
                value.append(tf * idf)
            else:
                value.append(cnt)

    index = torch.LongTensor(index)
    print(num_ent, num_time)
    matrix = torch.sparse_coo_tensor(torch.transpose(index, 0, 1), torch.Tensor(value),
                                     (num_ent, 2 * num_time))
    return matrix


def get_feature(filename):
    if filename == 'data/ICEWS05-15/':
        shift = 9517
        overlap = None
        time_id = None
    elif filename == 'data/YAGO-WIKI50K/':
        shift = 49629
        time_id = get_time(filename)
        print(len(time_id))
    else:
        shift = 19493
        time_id = get_time(filename)
        print(len(time_id))
    TF = False
    m1 = get_feature_matrix(filename, 1, 0, TF, time_id)
    m2 = get_feature_matrix(filename, 2, shift, TF, time_id)
    feature = torch.vstack((m1.to_dense(), m2.to_dense())).numpy()
    feature = tf.nn.l2_normalize(feature, axis=-1)
    feature = tf.cast(feature, tf.float64)
    return feature

def CSLS_test(thread_number = 16, csls=10,accurate = True):
    vec = embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    return None

def align_loss(align_input, embedding):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        # may not work
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))

    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]
    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    r_neg_dis = squared_dist([r_emb, embedding])
    l_neg_dis = squared_dist([l_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)
    # modified
    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std = torch.std(r_loss, dim=-1, keepdim=True)
        r_loss.data = (r_loss.data - r_mean) / r_std
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std = torch.std(l_loss, dim=-1, keepdim=True)
        l_loss.data = (l_loss.data - l_mean) / l_std

    lamb, tau = 30, 10
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)


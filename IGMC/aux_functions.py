import math

import numpy as np
import pandas as pd
import time
import tqdm
import dgl
import pickle as pkl
import numpy as np
import networkx as nx
from networkx import degree#导入networkx包
import matplotlib.pyplot as plt
import torch
import random
import os
import time
import torch.nn.functional as F
import dgl.function as fn
from torch.optim import Adam
from torch.utils.data import DataLoader
import multiprocessing as mp
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import Counter
import os.path
from urllib.request import urlopen
import scipy.sparse as sp
from zipfile import ZipFile
try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def map_data(data):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)
    return data, id_dict, n


def download_dataset(dataset, files, data_dir):
    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)
        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'raw_data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'raw_data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('raw_data/')

        os.rename(target_dir, data_dir)


def load_official_trainvaltest_split(dataset, testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    sep = '\t'

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'raw_data/' + fname

    download_dataset(fname, files, data_dir)

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    filename_train = 'raw_data/' + dataset + '/u1.base'
    filename_test = 'raw_data/' + dataset + '/u1.test'

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio *len(data_array_train))]]

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0: num_train +num_val]
    idx_nonzero_test = idx_nonzero[num_train +num_val:]

    pairs_nonzero_train = pairs_nonzero[0: num_train +num_val]
    pairs_nonzero_test = pairs_nonzero[num_train +num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if dataset == 'ml_100k':

        # movie features (genres)
        sep = r'|'
        movie_file = 'raw_data/' + dataset + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # user features

        sep = r'|'
        users_file = 'raw_data/' + dataset + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        age = users_df['age'].values
        age_max = age.max()

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    elif dataset == 'ml_1m':

        # load movie features
        movies_file = 'raw_data/' + dataset + '/movies.dat'

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # load user features
        users_file = 'raw_data/' + dataset + '/users.dat'
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    else:
        raise ValueError('Invalid dataset option %s' % dataset)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, num_users


def build_all_graph(adj_train, num_user, class_values):
    src = np.array(adj_train[0])
    dst = np.array(adj_train[1]) + num_user
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    G = dgl.DGLGraph((u, v))
    G.edata['rel_type'] = torch.tensor(np.concatenate([adj_train[2], adj_train[2]])).long()
    '''
    G.edata['rel_mod'] = torch.zeros(G.edata['rel_type'].shape[0], len(class_values))
    G.update_all(message_func, fn.sum(msg='msg', out='o'), apply_func)
    G.edata['norm'] = node_norm_to_edge_norm(G).reshape(-1,1)
    '''
    return G


class LocalGraphDataset(object):
    def __init__(self, G, links, labels, class_values, num_user, dataset=None, parallel=False, pre_save=False,
                 testing=False):
        self.G = G
        self.links = links
        self.data_len = len(self.links[0])
        self.labels = labels
        self.num_user = num_user
        self.testing = testing
        self.count = 0
        self.class_values = class_values
        self.pre_save = pre_save
        self.all_indexs = torch.arange(self.data_len)
        self.parallel = parallel
        self.dataset = dataset
        if self.pre_save:
            self.g_lists, self.labels = self.load_subgraphs(dataset)

    def len(self):
        return self.data_len

    def get_graph_tool(self, indexs):
        g_list = []
        labels = []
        index_out = []
        for index in indexs:
            g = self.extract_graph(self.G, self.links[0][index], self.links[1][index])
            index_out.append([self.links[0][index], self.links[1][index]])
            label = self.class_values[self.labels[index]]
            g_list.append(g)
            labels.append(label)
        return g_list, torch.FloatTensor(labels)

    def get_graph_tool_save(self, indexs):
        g_list = []
        labels = []
        index_out = []
        S = []
        if not self.parallel:
            pbar = tqdm(range(len(indexs)))
            for index in pbar:
                # g = self.extract_graph_new(self.G, self.links[0][index], self.links[1][index])
                # print(g.edges(form='all', order='srcdst')[0].shape)

                g = self.extract_graph(self.G, self.links[0][index], self.links[1][index])
                # print(g.edges(form='all', order='srcdst')[0].shape)
                # dd
                index_out.append([self.links[0][index], self.links[1][index]])
                # degree_l.append(g.number_of_edges())
                label = self.class_values[self.labels[index]]
                g_list.append(g)
                labels.append(label)
            return g_list, torch.FloatTensor(labels)
        else:
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(
                self.extract_graph,
                [
                    (self.G, self.links[0][index], self.links[1][index])
                    for index in indexs
                ]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            while results:
                tmp = results.pop()
                g_list.append(*tmp)
            labels = self.class_values[self.labels[indexs]]
            return g_list, torch.FloatTensor(labels)

    def get_graphs(self, indexs):
        if not self.pre_save:
            g_list, labels = self.get_graph_tool(indexs)
        else:
            g_list, labels = self.g_lists[indexs], self.labels[indexs]
        return dgl.batch(g_list), labels

    def extract_graph_n(self, G, u_id, v_id):
        subg = dgl.sampling.sample_neighbors(G, [u_id, v_id], 1)
        subg.ndata['node_label'] = torch.zeros([subg.num_nodes(), 4])
        pid = subg.ndata[dgl.NID]
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        if subg.has_edges_between(e_u, e_v):
            e_ids = subg.edge_ids([e_u, e_v], [e_v, e_u])
            subg.remove_edges(e_ids)
        return subg

    def save_subgraphs(self):
        if self.testing:
            file_path = "./data/" + self.dataset + "/test.bin"
            if os.path.exists(file_path):
                return
        else:
            file_path = "./data/" + self.dataset + "/train.bin"
            if os.path.exists(file_path):
                return
        g_list, labels = self.get_graph_tool_save(self.all_indexs)
        graph_labels = {"glabel": labels}
        save_graphs(file_path, g_list, graph_labels)

    def load_subgraphs(self):
        if self.testing:
            g_list, label_dict = load_graphs("./data/" + self.dataset + "/test/")
        else:
            g_list, label_dict = load_graphs("./data/" + self.dataset + "/train/")
        return g_list, label_dict["glabel"]

    def extract_graph_new(self, G, u_id, v_id):
        v_id += self.num_user
        static_u = torch.zeros(len(self.class_values))
        static_v = torch.zeros(len(self.class_values))
        start0 = time.time()
        u_nodes, v, e_ids_1 = G.in_edges(v_id, "all")
        u, v_nodes, e_ids_2 = G.out_edges(u_id, "all")
        e_ids = []
        nodes = torch.cat([u_nodes, v_nodes])
        for i in range(u_nodes.shape[0]):
            if u_nodes[i] == u_id:
                e_ids.append(e_ids_1[i])
        for i in range(v_nodes.shape[0]):
            if v_nodes[i] == v_id:
                e_ids.append(e_ids_2[i])
        # start1 = time.time()
        # print(start1 - start0)
        subg = dgl.node_subgraph(G, nodes)
        # start2 = time.time()
        # print(start2 - start1)
        subg.ndata['node_label'] = torch.zeros([subg.num_nodes(), 4])
        pid = subg.ndata[dgl.NID]
        # start3 = time.time()
        # print(start3 - start2)
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        subg = dgl.remove_edges(subg, e_ids)
        start6 = time.time()
        print(start6 - start0)
        print()
        return subg

    def extract_graph(self, G, u_id, v_id):
        v_id += self.num_user
        static_u = torch.zeros(len(self.class_values))
        static_v = torch.zeros(len(self.class_values))
        # start0 = time.time()
        u_nodes, v, e_ids = G.out_edges(u_id, "all")
        u, v_nodes, e_ids = G.in_edges(v_id, "all")
        nodes = torch.cat([u, v])
        if self.testing:
            nodes = torch.cat([nodes, torch.tensor([u_id, v_id])])
        # start1 = time.time()
        # print(start1 - start0)
        subg = G.subgraph(nodes)
        # start2 = time.time()
        # print(start2 - start1)
        subg.ndata['node_label'] = torch.zeros([nodes.shape[0], 4])
        pid = subg.ndata[dgl.NID]
        # start3 = time.time()
        # print(start3 - start2)
        for i in range(pid.shape[0]):
            if pid[i] == u_id:
                e_u = i
                subg.ndata['node_label'][i, 0] = 1
            elif pid[i] == v_id:
                e_v = i
                subg.ndata['node_label'][i, 1] = 1
            elif pid[i] in u:
                subg.ndata['node_label'][i, 2] = 1
            elif pid[i] in v:
                subg.ndata['node_label'][i, 3] = 1
        # start4 = time.time()
        # print(start4 - start3)
        if not self.testing:
            e_ids = subg.edge_ids([e_u, e_v], [e_v, e_u])
            # start5 = time.time()
            # print(start5 - start4)
            subg = dgl.remove_edges(subg, e_ids)
        # start6 = time.time()
        # print(start6 - start0)
        # print()
        return subg


def train_multiple_epochs(train_dataset,
                          test_dataset,
                          model,
                          epochs,
                          batch_size,
                          lr,
                          lr_decay_factor,
                          lr_decay_step_size,
                          weight_decay,
                          ARR = 0.0,
                          logger=None,
                          continue_from=None,
                          res_dir=None,
                          regression=True
                          ):
    seeds = torch.arange(train_dataset.len())
    train_loader = DataLoader(seeds, batch_size, collate_fn=train_dataset.get_graphs, shuffle=True, num_workers = mp.cpu_count())
    test_seeds = torch.arange(test_dataset.len())
    test_loader = DataLoader(test_seeds, batch_size,  collate_fn=test_dataset.get_graphs, shuffle=False, num_workers=mp.cpu_count())
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    rmses = []
    mrrs = []
    if continue_from is not None:
        model.load_state_dict(torch.load(os.path.join(res_dir, 'model_checkpoint{}.pth'.format(continue_from))))
        optimizer.load_state_dict(torch.load(os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(continue_from))))
        start_epoch = continue_from + 1
        epochs -= continue_from
    t_start = time.perf_counter()
    pbar = tqdm(range(start_epoch, epochs + start_epoch))
    best_mrr = 0.0
    print("start train:")
    test_pairs = test_dataset.links
    for epoch in pbar:
        t_start = time.perf_counter()
        train_loss = train(model, optimizer, train_loader, device, ARR = ARR)
        t_end = time.perf_counter()
        if regression:
            rmses.append(eval_rmse(model, test_loader, device))
        else:
            rmses.append(-1.0)
        #print(eval_rmse_dic(model, test_loader, device))
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
            'test_mrr': -1.0,
        }
        pbar.set_description('Epoch {}, train loss {:.6f}, test rmse {:.6f}, test mrrs {:.6f}'.format(*eval_info.values()))
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        duration = t_end - t_start
        print("The time in the %d epoch is %f" % (epoch, duration))
        if logger is not None:
            best_mrr = logger(eval_info, model, optimizer, best_mrr)


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False, verbose=True, rating_map=None, post_rating_map=None, ratio=1.0):
    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    if dataset == 'ml_100k':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    else:
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))

    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:int(num_train*ratio)]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    train_pairs_idx = pairs_nonzero[0:int(num_train*ratio)]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_data(fname, seed=1234, verbose=True):

    u_features = None
    v_features = None

    print('Loading dataset', fname)

    data_dir = 'raw_data/' + fname

    if fname == 'ml_100k':

        # Check if files exist and download otherwise
        files = ['/u.data', '/u.item', '/u.user']

        download_dataset(fname, files, data_dir)

        sep = '\t'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32, 'timestamp': np.float64}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

        # Movie features (genres)
        sep = r'|'
        movie_file = data_dir + files[1]
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # User features

        sep = r'|'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age']
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_1m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat', '/movies.dat', '/users.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

        # Load movie features
        movies_file = data_dir + files[1]

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # Extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # Creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # Load user features
        users_file = data_dir + files[2]
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # Extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_10m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

    else:
        raise ValueError('Dataset name not recognized: ' + fname)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features



def train(model, optimizer, loader, device, regression = True, ARR = 0.0):
    model.train()
    total_loss = 0
    t_start = time.perf_counter()
    for graphs, labels in loader:
        t_end = time.perf_counter()
        optimizer.zero_grad()
        #print(graphs.edges(form='all', order='srcdst'))
        graphs = graphs.to(device)
        labels = labels.to(device)
        out = model(graphs)
        if regression:
            loss = F.mse_loss(out, labels.view(-1))
        else:
            loss = F.nll_loss(F.log_softmax(out, dim = -1), labels.view(-1).long())
        if ARR != 0.0:
            for gconv in model.convs:
                w = torch.matmul(
                    gconv.w_comp,
                    gconv.weight.view(gconv.num_bases, -1)
                ).view(gconv.num_rels, gconv.in_feat, gconv.out_feat)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(out)
        optimizer.step()
        torch.cuda.empty_cache()
        t_start = time.perf_counter()
    return total_loss / len(loader.dataset)


def num_graphs(out):
    return out.shape[0]


def take_first(elem):
    return elem[0]


def eval_rmse(model, loader, device, show_progress=False):
    mse_loss = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


def eval_loss(model, loader, device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs)
        if regression:
            loss += F.mse_loss(out, labels.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, labels.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / len(loader.dataset)

def eval_rmse_dic(model, loader, device, show_progress=False):
    mse_loss_dic = eval_loss_dic(model, loader, device, True, show_progress)
    for item in mse_loss_dic.keys():
        mse_loss_dic[item] = math.sqrt(mse_loss_dic[item])
    mse_loss_dic = sorted(mse_loss_dic.items(), key= lambda d:d[1], reverse=False)
    return mse_loss_dic[-99:]


def eval_loss_dic(model, loader, device, regression=True, show_progress=False):
    model.eval()
    loss_dic = {}
    count_dic = {}
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs)
        if regression:
            loss = F.mse_loss(out, labels.view(-1), reduction='none')
        else:
            loss = F.nll_loss(out, labels.view(-1), reduction='none')
        y = degree.view(-1, 4).cpu().numpy().tolist()
        y2 = labels.view(-1).cpu().numpy().tolist()
        O = out.view(-1).cpu().numpy().tolist()
        l = loss.cpu().numpy().tolist()
        for i in range(len(y)):
            key = (y[i][0], y[i][1], y[i][2], y[i][3], y2[i], O[i])
            if key not in loss_dic:
                loss_dic[key] = l[i]
                count_dic[key] = 1
            else:
                loss_dic[key] += l[i]
                count_dic[key] += 1
        torch.cuda.empty_cache()
    for item in loss_dic.keys():
        loss_dic[item] /= count_dic[item]
    return loss_dic


def eval_mrr(model, loader, device, test_pairs, regression=True, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    ans = {}
    y = []
    hit_5 = 0
    hit_1 = 0
    hit_10 = 0
    count = 0
    debug = {}
    for i in range(len(test_pairs[0])):
        if test_pairs[0][i] not in debug:
            debug[test_pairs[0][i]] = [test_pairs[1][i]]
        else:
            debug[test_pairs[0][i]].append(test_pairs[1][i])
    for key in debug.keys():
        print(len(debug[key]))
    for graphs, labels in pbar:
        graphs = graphs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graphs).cpu().numpy()
            if not regression:
                out = out[:, 1]
            y = labels.cpu().numpy()
        for i in range(out.shape[0]):
            if test_pairs[0][count] in ans:
                ans[test_pairs[0][count]].append((out[i], y[i], test_pairs[1][count]))
            else:
                ans[test_pairs[0][count]] = [(out[i], y[i], test_pairs[1][count])]
            count += 1
        torch.cuda.empty_cache()
    c = []
    for key in ans.keys():
        ans[key].sort(key = take_first, reverse = True)
        print(str(key) + "_" + str(ans[key][0][2]))
        for i in range(len(ans[key])):
            if ans[key][i][1] == 1.0:
                c.append(1.0 / (i+1))
                if i < 1:
                    hit_1 += 1.0
                elif i < 5:
                    hit_5 += 1.0
                elif i < 10:
                    hit_10 += 1.0
    print("hit_1: " + str(hit_1 / len(c)))
    print("hit_5: " + str(hit_5 / len(c)))
    print("hit_10: " + str(hit_10 / len(c)))
    return sum(c) / len(c)

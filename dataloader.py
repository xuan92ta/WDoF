import os
import pandas as pd
import random
import numpy as np
from scipy import sparse


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=0, max_uc=0, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount[itemcount['size'] >= min_sc]['movidId'])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount[usercount['size'] >= min_uc]['userId'])]
        tp = tp[tp['userId'].isin(usercount[usercount['size'] <= max_uc]['userId'])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount

def split_support_query_proportion(data, support_size, unique_sid):
    data_grouped_by_user = data.groupby('userId')
    support_list, query_list = list(), list()
    support_list_neg, query_list_neg = list(), list()

    for i, (u, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        pos_set = set(group['movieId'].values)
        unwatched_set = set(unique_sid) - pos_set
        unwatched_array = np.array(list(unwatched_set))

        if n_items_u >= support_size:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=support_size, replace=False).astype('int64')] = True

            support_list.append(group[idx])
            query_list.append(group[np.logical_not(idx)])

            # 选择同等数量的负样本
            idx_neg_supp = np.random.choice(len(unwatched_array), size=support_size, replace=False).astype('int64')
            idx_neg_query = np.random.choice(len(unwatched_array), size=n_items_u - support_size, replace=False).astype('int64')

            pd_supp_neg = pd.DataFrame({'userId': [], 'movieId': []})
            pd_query_neg = pd.DataFrame({'userId': [], 'movieId': []})

            for neg_i in unwatched_array[idx_neg_supp]:
                pd_supp_neg.loc[len(pd_supp_neg)] = {'userId': u, 'movieId': neg_i}
            for neg_i in unwatched_array[idx_neg_query]:
                pd_query_neg.loc[len(pd_query_neg)] = {'userId': u, 'movieId': neg_i}
            
            support_list_neg.append(pd_supp_neg)
            query_list_neg.append(pd_query_neg)

    data_support = pd.concat(support_list)
    data_query = pd.concat(query_list)
    data_support_neg = pd.concat(support_list_neg)
    data_query_neg = pd.concat(query_list_neg)
    
    return data_support, data_query, data_support_neg, data_query_neg

def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def dataloader(opt):
    save_path = os.path.join(os.path.join(opt['data_dir'], opt['dataset']), 'pro_sg')

    print(save_path)

    # If preprocessed data does not exist, reprocess and store it; otherwise, read the data.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        if opt['dataset'] == 'lastfm':
            raw_data = pd.read_csv(os.path.join(os.path.join(opt['data_dir'], opt['dataset']), 'ratings_final.txt'), header=None, names=['userId', 'movieId', 'rating'], sep='\t')
            raw_data = raw_data[raw_data['rating'] == 1]
        elif opt['dataset'] == 'ml1m':
            raw_data = pd.read_csv(os.path.join(os.path.join(opt['data_dir'], opt['dataset']), 'ratings.dat'), header=None, names=['userId', 'movieId', 'rating', 'timestamp'], sep='::')
            raw_data = raw_data[raw_data['rating'] > 3.5]
        elif opt['dataset'] == 'epinions':
            raw_data = pd.read_csv(os.path.join(os.path.join(opt['data_dir'], opt['dataset']), 'ratings_data.txt'), header=None, names=['userId', 'movieId', 'rating'], sep=' ')
            raw_data = raw_data[raw_data['rating'] > 3.5]
        elif opt['dataset'] == 'yelp':
            raw_data = pd.read_csv(os.path.join(os.path.join(opt['data_dir'], opt['dataset']), 'Yelp_ratings.csv'), header=0, names=['userId', 'movieId', 'rating'], sep=',')
            raw_data = raw_data[raw_data['rating'] > 3.5]
        
        # Split the train/validation/test sets according to users.
        unique_uid = raw_data['userId'].unique()
        random.shuffle(unique_uid)

        n_users = len(unique_uid)
        n_tr_users = int(n_users * opt['train_ratio'])
        n_vd_users = int(n_users * opt['valid_ratio'])

        tr_users = unique_uid[: n_tr_users]
        vd_users = unique_uid[n_tr_users: n_tr_users + n_vd_users]
        te_users = unique_uid[n_tr_users + n_vd_users:]

        assert n_users == len(tr_users) + len(vd_users) + len(te_users)

        # Only use items that have appeared in the training set for testing.
        train_data = raw_data.loc[raw_data['userId'].isin(tr_users)]
        unique_sid = pd.unique(train_data['movieId'])
        raw_data = raw_data.loc[raw_data['movieId'].isin(unique_sid)]

        # Only retain users with interactions in the range of [15,100].
        # There might be a situation where test items have not appeared in the training set, but we'll disregard that for now.
        raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=15, min_sc=0, max_uc=100)

        sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
        print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
            (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
        
        # Update user id and train/validation/test sets.
        unique_uid= user_activity['userId']
        tr_users = np.array(list(set(tr_users).intersection(unique_uid)))
        vd_users = np.array(list(set(vd_users).intersection(unique_uid)))
        te_users = np.array(list(set(te_users).intersection(unique_uid)))

        # Split support/query set.
        train_data = raw_data.loc[raw_data['userId'].isin(tr_users)]
        valid_data = raw_data.loc[raw_data['userId'].isin(vd_users)]
        test_data = raw_data.loc[raw_data['userId'].isin(te_users)]

        unique_sid_train = pd.unique(train_data['movieId'])

        train_data_support, train_data_query, train_data_support_neg, train_data_query_neg = split_support_query_proportion(train_data, opt['support_size'], unique_sid_train)
        valid_data_support, valid_data_query, valid_data_support_neg, valid_data_query_neg = split_support_query_proportion(valid_data, opt['support_size'], unique_sid_train)
        test_data_support, test_data_query, test_data_support_neg, test_data_query_neg = split_support_query_proportion(test_data, opt['support_size'], unique_sid_train)

        # Save user/item id mapping.
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

        # Store the results of data preprocessing.
        train_data_support = numerize(train_data_support, profile2id, show2id)
        train_data_support.to_csv(os.path.join(save_path, 'train_support.csv'), index=False)
        train_data_query = numerize(train_data_query, profile2id, show2id)
        train_data_query.to_csv(os.path.join(save_path, 'train_query.csv'), index=False)
        valid_data_support = numerize(valid_data_support, profile2id, show2id)
        valid_data_support.to_csv(os.path.join(save_path, 'valid_support.csv'), index=False)
        valid_data_query = numerize(valid_data_query, profile2id, show2id)
        valid_data_query.to_csv(os.path.join(save_path, 'valid_query.csv'), index=False)
        test_data_support = numerize(test_data_support, profile2id, show2id)
        test_data_support.to_csv(os.path.join(save_path, 'test_support.csv'), index=False)
        test_data_query = numerize(test_data_query, profile2id, show2id)
        test_data_query.to_csv(os.path.join(save_path, 'test_query.csv'), index=False)

        train_data_support_neg = numerize(train_data_support_neg, profile2id, show2id)
        train_data_support_neg.to_csv(os.path.join(save_path, 'train_support_neg.csv'), index=False)
        train_data_query_neg = numerize(train_data_query_neg, profile2id, show2id)
        train_data_query_neg.to_csv(os.path.join(save_path, 'train_query_neg.csv'), index=False)
        valid_data_support_neg = numerize(valid_data_support_neg, profile2id, show2id)
        valid_data_support_neg.to_csv(os.path.join(save_path, 'valid_support_neg.csv'), index=False)
        valid_data_query_neg = numerize(valid_data_query_neg, profile2id, show2id)
        valid_data_query_neg.to_csv(os.path.join(save_path, 'valid_query_neg.csv'), index=False)
        test_data_support_neg = numerize(test_data_support_neg, profile2id, show2id)
        test_data_support_neg.to_csv(os.path.join(save_path, 'test_support_neg.csv'), index=False)
        test_data_query_neg = numerize(test_data_query_neg, profile2id, show2id)
        test_data_query_neg.to_csv(os.path.join(save_path, 'test_query_neg.csv'), index=False)

        with open(os.path.join(save_path, 'user_dict.txt'), 'w') as f:
            f.write("ori_id,new_id\n")
            for k, v in profile2id.items():
                f.write(str(k) + "," + str(v) + "\n")
        with open(os.path.join(save_path, 'item_dict.txt'), 'w') as f:
            f.write("ori_id,new_id\n")
            for k, v in show2id.items():
                f.write(str(k) + "," + str(v) + "\n")
        with open(os.path.join(save_path, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % profile2id[uid])
        with open(os.path.join(save_path, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % show2id[sid])
        
    # If preprocessed data already exists, load the data.
    unique_sid = list()
    with open(os.path.join(save_path, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    train_data_support, train_data_query = load_tr_te_data(os.path.join(save_path, 'train_support.csv'),
                                        os.path.join(save_path, 'train_query.csv'), n_items)
    valid_data_support, valid_data_query = load_tr_te_data(os.path.join(save_path, 'valid_support.csv'),
                                        os.path.join(save_path, 'valid_query.csv'), n_items)
    test_data_support, test_data_query = load_tr_te_data(os.path.join(save_path, 'test_support.csv'),
                                        os.path.join(save_path, 'test_query.csv'), n_items)
    
    train_data_support_neg, train_data_query_neg = load_tr_te_data(os.path.join(save_path, 'train_support_neg.csv'),
                                        os.path.join(save_path, 'train_query_neg.csv'), n_items)
    valid_data_support_neg, valid_data_query_neg = load_tr_te_data(os.path.join(save_path, 'valid_support_neg.csv'),
                                        os.path.join(save_path, 'valid_query_neg.csv'), n_items)
    test_data_support_neg, test_data_query_neg = load_tr_te_data(os.path.join(save_path, 'test_support_neg.csv'),
                                        os.path.join(save_path, 'test_query_neg.csv'), n_items)
    

    print("dataloader is done :)")
    return train_data_support, train_data_query, valid_data_support, valid_data_query, test_data_support, test_data_query, train_data_support_neg, train_data_query_neg, valid_data_support_neg, valid_data_query_neg, test_data_support_neg, test_data_query_neg, n_items
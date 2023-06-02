import os
import torch
import argparse
import tqdm
import random
from scipy import sparse
import numpy as np
from datetime import datetime

from dataloader import dataloader
from model import WDoF
from scorer import add_metric
import helper


def seed_everything(seed=1023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(opt, model, train_sup, train_que, valid_sup, valid_que, valid_sup_neg, valid_que_neg, logger, log_path, log_name):
    # Reconstruct the query set to include the support set.
    train_que = train_sup + train_que
    
    n_train_users = train_sup.shape[0]
    idxlist = list(range(n_train_users))
    batch_size = opt['batch_size']

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])

    best_ndcg = 0
    best_epoch = None

    loss_train_record = []
    ndcg_record = []

    epoch_iterator = tqdm.tqdm(range(opt['n_epoch']))

    # training
    for epoch in epoch_iterator:
        if opt['shuffle']:
            random.shuffle(idxlist)

        model.train()

        loss_list_train = []
        neg_ll_list_train = []
        KL_list_train = []

        for bnum, st_idx in enumerate(range(0, n_train_users, batch_size)):
            end_idx = min(st_idx + batch_size, n_train_users)
            X_sup = train_sup[idxlist[st_idx:end_idx]]
            X_que = train_que[idxlist[st_idx:end_idx]]

            n_inter = X_que.sum(axis=1)
            weight = np.sin((np.pi / 2) * (epoch / opt['n_epoch']) + (np.pi / 2) * (n_inter - opt['l_min']) / (opt['l_max'] - opt['l_min']))  # 最大交互数目为100，最小交互数目为15
            weight = torch.tensor(weight).squeeze(dim=-1)
            weight[weight < 0] = 0
            weight = weight.cuda()

            if sparse.isspmatrix(X_sup):
                X_sup = X_sup.toarray()
                X_que = X_que.toarray()
            X_sup = torch.tensor(X_sup.astype('float32'))
            X_que = torch.tensor(X_que.astype('float32'))

            X_sup_cuda = X_sup.cuda()
            X_que_cuda = X_que.cuda()

            _, neg_ll, KL, loss = model(X_sup_cuda, X_que_cuda, weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list_train.append(loss.item())
            neg_ll_list_train.append(neg_ll.item())
            KL_list_train.append(KL.item())

        total_loss_train = np.mean(np.array(loss_list_train))
        total_neg_ll_train = np.mean(np.array(neg_ll_list_train))
        total_KL_train = np.mean(np.array(KL_list_train))

        loss_train_record.append(total_loss_train.item())

        logger.log("Epoch %d: \t loss = neg_ll + KL: %.4f = %.4f + %.4f" % (epoch, total_loss_train, total_neg_ll_train, total_KL_train))
        
        
        # validation
        total_ndcg_7 = test(model, valid_sup, valid_que, valid_que_neg, logger)
            
        ndcg_record.append(total_ndcg_7)

        if total_ndcg_7 > best_ndcg:
            best_ndcg = total_ndcg_7
            best_epoch = epoch

            if opt['save']:
                torch.save(model.state_dict(), os.path.join(log_path, log_name) + '.pt')
        
        desc = ('Epoch {epoch:d}: Loss={loss:.4f} | '
                'ndcg@7={ndcg:.4f} | Best_ndcg={best_ndcg:.4f}'.format( 
                epoch=epoch, loss=total_loss_train, ndcg=total_ndcg_7, best_ndcg=best_ndcg))
        epoch_iterator.set_description(desc)
        epoch_iterator.refresh()

    logger.log("In validation phase, the best epoch is %d, and the corresponding ndcg@7 = %.4f." % (best_epoch, best_ndcg))
    
    return 

def test(model, test_sup, test_que, test_que_neg, logger):
    ndcg_list_5, ndcg_list_7, ndcg_list_10 = [], [], []
    precision_list_5, precision_list_7, precision_list_10 = [], [], []

    model.eval()

    with torch.no_grad():
        test_users = np.count_nonzero(test_que.toarray(), axis=1).nonzero()[0]
        for u in test_users:
            X = test_sup[u]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X.astype('float32'))

            X_cuda = X.cuda()
            
            logit, _, _, _ = model(X_cuda)

            logit = logit.cpu().detach().numpy()
            logit[X.numpy().nonzero()] = -np.inf
        
            test_que_pos_list = list(test_que[u].nonzero()[1])  # the id of positive feedback in query set
            test_que_neg_list = list(test_que_neg[u].nonzero()[1])  # the id of negative feedback in query set
            test_que_like_tanp = np.concatenate([test_que_pos_list, test_que_neg_list])

            recommendation_list = np.argsort(-logit[:, test_que_like_tanp])
            add_metric(recommendation_list[0], test_que[u][:, test_que_like_tanp].toarray()[0], precision_list_5, ndcg_list_5, 5)
            add_metric(recommendation_list[0], test_que[u][:, test_que_like_tanp].toarray()[0], precision_list_7, ndcg_list_7, 7)
            add_metric(recommendation_list[0], test_que[u][:, test_que_like_tanp].toarray()[0], precision_list_10, ndcg_list_10, 10)
           
        total_ndcg_5 = np.mean(ndcg_list_5)
        total_ndcg_7 = np.mean(ndcg_list_7)
        total_ndcg_10 = np.mean(ndcg_list_10)
        total_precision_5 = np.mean(precision_list_5)
        total_precision_7 = np.mean(precision_list_7)
        total_precision_10 = np.mean(precision_list_10)

        logger.log("[validation]: ndcg@5 = %.4f, ndcg@7 = %.4f, ndcg@10 = %.4f, pre@5 = %.4f, pre@7 = %.4f, pre@10 = %.4f" % (total_ndcg_5, total_ndcg_7, total_ndcg_10, total_precision_5, total_precision_7, total_precision_10))
    return total_ndcg_7
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lastfm', choices=['lastfm', 'ml1m', 'epinions', 'yelp'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--log_name', type=str, default='WDoF')

    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='User ratio for training.')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='User ratio for validation.')
    parser.add_argument('--support_size', type=int, default=10)

    parser.add_argument('--l_max', type=int, default=100, help='The maximum number of interactions.')
    parser.add_argument('--l_min', type=int, default=0, help='The minimal number of interactions.')

    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--shuffle', type=bool, default=False)

    parser.add_argument('--z_dim', type=int, default=200, help='The dimension of z in latent path.')
    parser.add_argument('--enc_dim', nargs='?', default='[600]', help='The hidden dimension of encoder.')
    parser.add_argument('--dec_dim', nargs='?', default='[600]', help='The hidden dimension of decoder.')

    parser.add_argument('--save', type=int, default=0)

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_name', type=str, default='')

    args = parser.parse_args()
    opt = vars(args)

    seed_everything(seed=args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    torch.set_num_threads(1)

    log_path = os.path.join(opt['log_path'], opt['dataset'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    now = datetime.now()
    log_name = opt['log_name'] + "_" + now.strftime("%m%d_%H%M")
    # print model log
    helper.print_config(opt)
    helper.save_config(opt, os.path.join(log_path, log_name) + '.config', verbose=True)
    # record training log
    file_logger = helper.FileLogger(os.path.join(log_path, log_name) + ".log")

    # dataloader
    train_sup, train_que, valid_sup, valid_que, test_sup, test_que, train_sup_neg, train_que_neg, valid_sup_neg, valid_que_neg, test_sup_neg, test_que_neg, n_items = dataloader(opt)

    seed_everything(seed=args.seed)
    model = WDoF(opt, n_items)
    model = model.cuda()

    if not opt['pretrain']:
        print("start training :)")
        train(opt, model, train_sup, train_que, test_sup, test_que, test_sup_neg, test_que_neg, file_logger, log_path, log_name)
    else:
        trained_state_dict = torch.load("{}/{}.pt".format(log_path, "WDoF" + opt['pretrain_name']))
        model.load_state_dict(trained_state_dict)

        ndcg7 = test(model, test_sup, test_que, test_sup_neg, test_que_neg, file_logger, opt['test_all'])
        print("In test phase, the corresponding ndcg@7 = %.4f." % (ndcg7))
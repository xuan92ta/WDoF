import math

def precision(ranked_list,ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(min(len(t), topn)):
        id = ranked_list[i]
        if ground_truth[id] in t and ground_truth[id] == 1:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre

def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    for i in range(min(len(ground_truth), topn)):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) -1)/ math.log(i+2, 2)
    return dcg / idcg

def IDCG(ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(min(len(t), topn)):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def add_metric(recommend_list, ALL_group_list, precision_list, ndcg_list, topn):
    ndcg = nDCG(recommend_list, ALL_group_list, topn)
    pre = precision(recommend_list, ALL_group_list, topn)
    precision_list.append(pre)
    ndcg_list.append(ndcg)


def cal_metric(precision_list,ap_list,ndcg_list):
    mpre = sum(precision_list) / len(precision_list)
    map = sum(ap_list) / len(ap_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return mpre, mndcg, map

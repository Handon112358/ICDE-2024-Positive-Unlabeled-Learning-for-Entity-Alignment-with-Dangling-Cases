import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing

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
    max_val=0
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
        
        max_val=max(max_val,head,tail)
        
    return entity,rel,triples,max_val

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples1,triples2,entity,rel):
    ent_size = max(entity)+1
    rel_size = (max(rel) + 1)
    print(ent_size,rel_size)
    adj_matrix = sp.lil_matrix((ent_size,ent_size))
    adj_features = sp.lil_matrix((ent_size,ent_size))
    radj = []
    rel_in = np.zeros((ent_size,rel_size))
    rel_out = np.zeros((ent_size,rel_size))

    for i in range(max(entity)+1):
        adj_features[i,i] = 1
        adj_matrix[i,i]=1

    triples = triples1 + triples2

    for h,r,t in triples: #"edge doubling"
        adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
        adj_features[h,t] = 1; adj_features[t,h] = 1;
        radj.append([h,t,r]); radj.append([t,h,r+rel_size]);
        rel_out[h][r] += 1; rel_in[t][r] += 1

    idx_limit = len(triples1)
    origin_index = [i for i in range(idx_limit*2)]
    original_map = {i: triple for i, triple in enumerate(radj)}#原始映射
    sorted_radj = sorted(radj, key=lambda x: x[0]*10e10+x[1]*10e5)
    new_map = {(triple[0],triple[1],triple[2]):i for i, triple in enumerate(sorted_radj)}#新的映射
    original_map_triple = []
    for i in origin_index:
        original_map_triple.append(original_map[i])
    new_indexes = []
    new_indices = []
    for triple in original_map_triple:
        new_indexes.append([new_map[(triple[0],triple[1],triple[2])]])
        new_indices.append(new_map[(triple[0],triple[1],triple[2])])

    count = -1
    s = set()
    d = {}
    r_index,r_val = [],[]
    for h,t,r in sorted_radj:#对于两个实体，他们之间所有的关系都要集中处理
        if ' '.join([str(h),str(t)]) in s:
            r_index.append([count,r])#d代表的是当前两个实体对的编号为count,他们之间有d[count]个关系存在
            r_val.append(1)
            d[count] += 1#两个节点之间的边的数目
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h),str(t)]))
            r_index.append([count,r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]#正则化
    rel_features = np.concatenate([rel_in,rel_out],axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix,r_index,r_val,adj_features,rel_features,[new_indexes,new_indices]

    
def load_data(lang,train_ratio = 0.3):             
    entity1,rel1,triples1,max_val1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2,max_val2 = load_triples(lang + 'triples_2')
    if "_en" in lang:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        np.random.shuffle(alignment_pair)
        train_pair,dev_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)],alignment_pair[int(len(alignment_pair)*train_ratio):]
    else:
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
        ae_features = None
    adj_matrix,r_index,r_val,adj_features,rel_features,new_indexes = get_matrix(triples1,triples2,entity1.union(entity2),rel1.union(rel2))
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features,new_indexes,max_val1,max_val2

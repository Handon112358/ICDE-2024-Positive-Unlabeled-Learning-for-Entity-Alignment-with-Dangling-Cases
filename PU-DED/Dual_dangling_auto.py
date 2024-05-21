import warnings
warnings.filterwarnings('ignore') 

import os
import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
from evaluate import evaluate

import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import NR_GraphAttention
from layer import tar_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# fig_save='data/subgraph/fr_fr/'
# fig_save="data/16k_update/"
# fig_save='data/zh_en/'
# fig_save='data/dbp2.0/fr_en/'
# fig_save='data/dbp2.0/zh_en/'
# fig_save='data/MedED/es/'
# fig_save='data/GA-XX/GA-FR/0.15_en/'
# fig_save='data/GA-XX/GA-JA/0.15_en/'
# fig_save='data/GA-XX/GA-ZH/0.15_en/'
# fig_save='data/GA-XX/GA-EN/0.15_en/'
# fig_save='data/hyh_prp/0.75_en/'
# fig_save='data/MedED/fr_en/'
# fig_save='data/dbp2.0/ja_en/'
# fig_save='data/dbp2.0/fr_en/'
# fig_save="data/16k_update/"
# fig_save='data/subgraph/fr_fr/'
# fig_save="data/16k_update/"
# fig_save='data/dbp_pm/plus/DBP2.0-JA-EN-plus_en/'
# fig_save='data/dbp_pm/plus/DBP2.0-ZH-EN-plus_en/'
# fig_save='data/dbp_pm/minus/DBP2.0-JA-EN-minus_en/'
fig_save='data/dbp_pm/minus/DBP2.0-ZH-EN-minus_en/'

train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true,max_val1,max_val2 = load_data(fig_save,train_ratio=0.3)
# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true,max_val1,max_val2 = load_data(fig_save,train_ratio=0.99)

print("max_val1")
print(max_val1)
print("max_val2")
print(max_val2)

# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,new_indexes,y_true = load_data(fig_save,train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)

node_hidden = 96
rel_hidden = 96
batch_size = 5120#
dropout_rate = 0.3#
lr = 0.005#
gamma = 1#
depth = 2#
tar=int(node_size*0.75)

print("triples:")
print(triple_size)
print("node:")
print(node_size)
print("rel:")
print(rel_size)

indices_ = np.concatenate((train_pair[:,0], train_pair[:,1]), axis=0)
p_size = len(indices_)
print("p_size")
print(p_size)

#unlabeled sample
un_indices = []
for i in range(node_size):
    if i not in indices_:   
        un_indices.append(i)
un_indices = np.array(un_indices)
u_size = len(un_indices)

print("u_size")
print(u_size)

anchor_size = len(train_pair)
print("anchor_size")
print(anchor_size)
''''''

global tar_tensor
tar_tensor = tf.zeros([node_size], dtype=tf.float32)
length=len(target_dis_tag)
##debug
print("length=",length)

indices=[]
for i in target_dis_tag:
    indices.append([i])
indices = tf.constant(indices)
updates = tf.constant([1.0]*length)
tar_tensor=tf.tensor_scatter_nd_update(tar_tensor, indices, updates)


#========================基本属性只需要导入一次=========================#

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings

'''分类模型'''
def get_embedding(index_a, index_b, vec=None):
    if vec is None:
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        vec, out_logits = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec, out_logits#, loss

def get_trgat(new_indexes, node_hidden, rel_hidden, triple_size=triple_size, node_size=node_size, rel_size=rel_size, p_size=p_size, u_size=u_size,anchor_size=anchor_size, dropout_rate=0, gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))#1
    index_input = Input(shape=(None, 2), dtype='int64')#2
    val_input = Input(shape=(None,))#3
    rel_adj = Input(shape=(None, 2))#4
    ent_adj = Input(shape=(None, 2))#5

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    e_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size,
                                  tar=tar,
                                  tar_tensor=tar_tensor,
                                  new_indexes=new_indexes,
                                  name = 'e_encoder')
    
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])
    r_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size,
                                  tar=tar,
                                  tar_tensor=tar_tensor,
                                  new_indexes=new_indexes,
                                  name = 'r_encoder')
    
    [e_feature,d_e] = e_encoder([ent_feature] + opt)#p_e注意力分布
    [r_feature,d_r] = r_encoder([rel_feature] + opt)

    
    e_out_feature = Concatenate(-1)([e_feature, d_e])#先进行e
    r_out_feature = Concatenate(-1)([r_feature, d_r])
    out_feature = Concatenate(-1)([e_out_feature, r_out_feature])#最后合并# out_feature = Concatenate(-1)([e_feature, r_feature])#最后合并
    # r_out_feature = Concatenate(-1)([e_feature, d_e])#再进行r
    # out_feature = e_out_feature
    
    out_feature = Dropout(dropout_rate)(out_feature)
    out_logits = Dense(2,activation='softmax')(out_feature)
    
    alignment_input = Input(shape=(None, 2))#6
    y_true = Input(shape=(None,2))#7
    
    indices = Input(shape=(None,), dtype='int32')#8
    un_indices = Input(shape=(None,),dtype='int32')#9
    pi_p   = Input(shape=(None,),dtype='float32')
    pi_p_u = Input(shape=(None,),dtype='float32')
    
    def data_y_p_p_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        n = p_size
        y_train = tf.constant([[1.0, 0.0]] * n)
        return y_train

    def data_y_p_n_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        n = p_size
        y_train = tf.constant([[0.0, 1.0]] * n)
        return y_train
    
    def data_y_u_n_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        n = u_size
        y_train = tf.constant([[0.0, 1.0]] * n)
        return y_train
    
    def data_out_logits(tensor):
        out_logits = tensor[0]
        indices = tensor[1]
        out_logits_train = K.gather(out_logits, indices)
        return out_logits_train
    
    y_p_p = Lambda(data_y_p_p_label)([y_true, indices])
    y_p_n = Lambda(data_y_p_n_label)([y_true, indices])
    y_u_n = Lambda(data_y_u_n_label)([y_true, un_indices])
    
    logits_y_p = Lambda(data_out_logits)([out_logits, indices])
    logits_y_u = Lambda(data_out_logits)([out_logits, un_indices])

    def align_loss(tensor):
        def squared_dist(x):
            A, B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)
        
        emb = tensor[1]
        l, r = K.cast(tensor[0][0, :, 0], 'int32'), K.cast(tensor[0][0, :, 1], 'int32')
        l_emb, r_emb = K.gather(reference=emb, indices=l), K.gather(reference=emb, indices=r)
        
        '''SNNL'''
        '''SNNL'''

        pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)
        r_neg_dis = squared_dist([r_emb, emb])
        l_neg_dis = squared_dist([l_emb, emb])

        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(r_loss, axis=-1, keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(l_loss, axis=-1, keepdims=True))

        lamb, tau = 30, 10
        # keta = 0.5
        l_loss = K.logsumexp(lamb * l_loss + tau, axis=-1)
        r_loss = K.logsumexp(lamb * r_loss + tau, axis=-1)
        return K.mean(l_loss + r_loss)

    loss_EA = Lambda(align_loss)([alignment_input, out_feature])
    
    def Loss(tensor):
        y_p_p = tensor[1]
        y_p_n = tensor[2]
        y_u_n = tensor[3]
        logits_y_p = K.squeeze(tensor[4], axis=0)
        logits_y_u = K.squeeze(tensor[5], axis=0)
        
        CE_p_p = -K.mean(K.sum(K.reshape(y_p_p * tf.log(logits_y_p), (p_size, 2)), axis=1))
        CE_p_n = -K.mean(K.sum(K.reshape(y_p_n * tf.log(logits_y_p), (p_size, 2)), axis=1))
        CE_u_n = -K.mean(K.sum(K.reshape(y_u_n * tf.log(logits_y_u), (u_size, 2)), axis=1))

        # 最优dangling
        # beta = 6
        # pi_p = beta*tf.cast(K.shape(y_p_p)[0], tf.float32) / (tf.cast(K.shape(y_u_n)[0], tf.float32))
        # pi_p_p = pi_p*0.3
        # pi_p_n = pi_p*0.7
        # CE_nn = K.relu(CE_u_n - pi_p_n*CE_p_n)#max
        # gama = 1
        # return gama*tensor[0] + (1 - gama) * (pi_p_p * CE_p_p + CE_nn)
        # 最优matchable
        
        # beta = 3.0  2.5
        # beta = 3.0
        # beta = 1.0
        
        # pi_p = beta*tf.cast(K.shape(y_p_p)[0], tf.float32) / ((tf.cast(K.shape(y_u_n)[0], tf.float32))+(tf.cast(K.shape(y_p_p)[0], tf.float32)))
        pi_p = tf.cast(tensor[6], tf.float32)
        pi_p_u = tf.cast(tensor[7], tf.float32)
        # beta = 3.0
        alpha = (1 - pi_p_u)/(1 - pi_p)
        # print(pi_p)
        # print(pi_p_u)
        # print(alpha)
        # CE_nn = K.relu(CE_u_n - pi_p*CE_p_n)
        CE_nn = K.relu(CE_u_n - pi_p_u * CE_p_n)
        gama = 0.0
        
        # return gama*tensor[0] + (1 - gama) * (pi_p * CE_p_p + CE_nn)
        return gama * tensor[0] + (1 - gama) * (alpha * pi_p * CE_p_p + CE_nn)
    
    loss = Lambda(Loss)([loss_EA, y_p_p, y_p_n, y_u_n, logits_y_p, logits_y_u, pi_p, pi_p_u])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input, y_true, indices, un_indices, pi_p, pi_p_u], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))
    
    # feature_model = keras.Model(inputs=inputs, outputs=[out_feature,final_reweight,out_logits])
    feature_model = keras.Model(inputs=inputs, outputs=[out_feature,out_logits])
    return train_model, feature_model

# node_hidden_=[96]#4
node_hidden_=[32]#4
# node_hidden_=[16]#4
depth_ = [2]#2
ratio_=[0.75]
dropout_rate_=[0.3]
flag=1

def Pred_logits(out_logits,threshold=0.5):
    pred_labels = np.where(out_logits[:, 0] >= threshold, 0, 1)
    return pred_labels

pair = dev_pair.tolist() + train_pair.tolist()
pred_labels=[]

def test_dev_dev():
    '''test'''
    
    Lvec, Rvec, out_logits = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
    print("a")
    evaluater = evaluate(dev_pair,len(dev_pair))
    print("b")
    evaluater.test(Lvec, Rvec)
    print("c")
    return out_logits

def test_dev_all(rest_set_1,rest_set_2,precision1,recall1,precision2,recall2):
    '''test'''
    Lvec, Rvec, out_logits = get_embedding(dev_pair[:, 0], rest_set_2)
    evaluater1 = evaluate(dev_pair,len(rest_set_2))
    hits1, hits5, hits10, mrr = evaluater1.test_real(Lvec, Rvec)
    Prec = hits1*precision1
    Rec = hits1*recall1
    if Prec + Rec != 0:
        F1 = 2 * Prec * Rec / (Prec + Rec)
    else:
        F1=0
    print('Prec',Prec,'Rec',Rec,'F1',F1)
    
    Lvec, Rvec, out_logits = get_embedding(rest_set_1, dev_pair[:, 1])
    evaluater2 = evaluate(dev_pair,len(rest_set_1))
    hits1, hits5, hits10, mrr = evaluater2.test_real(Rvec, Lvec)
    Prec = hits1*precision2
    Rec = hits1*recall2
    if Prec + Rec != 0:
        F1 = 2 * Prec * Rec / (Prec + Rec)
    else:
        F1=0
    print('Prec',Prec,'Rec',Rec,'F1',F1)
    
    return out_logits

def test_all_all():
    return 

def classify_node_size(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(node_size):#test and train
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FN+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TN+=1
            continue

    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)

    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:   
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)
    if precision + recall!=0:
        print("node_size", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)    
    return precision,recall

def classify_test_node(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # train_node=[i[0] for i in pair]#原本图上的标记为绿色，看看绿色和红色，其他没用的标记为黑色
    test_node=rest_set_1+rest_set_2#kg1和kg2上面的总测试集
    for i in test_node:
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FN+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TN+=1
            continue
    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:    
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)        

    if precision + recall!=0:
        print("test_node", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    return precision,recall 

def classify_test_node_1(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_node=rest_set_1
    # +dev_pair[:, 1].tolist()#kg1和kg2上面的总测试集
    for i in test_node:
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FN+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TN+=1
            continue
    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:    
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)        

    if precision + recall!=0:
        print("test_node_1", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    return precision,recall

def classify_test_node_2(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_node=rest_set_2
    # +dev_pair[:, 0].tolist()#kg1和kg2上面的总测试集
    for i in test_node:
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FN+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TN+=1
            continue
    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:    
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)        

    if precision + recall!=0:
        print("test_node_2", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    return precision,recall

def classify_test_node_1_rec(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_node=rest_set_1
    # +dev_pair[:, 0].tolist()#kg1和kg2上面的总测试集
    for i in test_node:
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FN+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TN+=1
            continue
    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:    
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)        

    if precision + recall!=0:
        print("test_node_2", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    return precision,recall

def classify_test_node_2_rec(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_node = rest_set_2
    for i in test_node:
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FN+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TN+=1
            continue
    print("TP", TP, "FP", FP, "FN", FN, "TN", TN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = 0
    if TP + FP!=0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN!=0:
        recall = TP /(TP + FN)
    F1 = 0
    F2 = 0
    if precision + recall!=0:    
        F1 = 2 * precision * recall / (precision + recall)
        beta = 2
        F_beta = (1+beta*beta)*precision * recall / (beta*beta*precision + recall)        

    if precision + recall!=0:
        print("test_node_2", "Accuracy", Accuracy,  
              "precision", precision,
              "recall", recall,       
              "F1", F1,        
              "F2", F_beta)
    print("--------------------------")
    return precision,recall

rest_set_1 = [e1 for e1, e2 in dev_pair]#test
rest_set_2 = [e2 for e1, e2 in dev_pair]#test
dev_set = set(dev_pair[:, 0]) | set(dev_pair[:, 1])
dev_len = len(dev_set)

train_set_1 = set(train_pair[:, 0])
train_set_2 = set(train_pair[:, 1])
train_set = train_set_1 | train_set_2
train_len = len(train_set)

train_list_1 = [e1 for e1, e2 in train_pair]
train_list_2 = [e2 for e1, e2 in train_pair]

rest_set_1_add = set(i for i in range(0,max_val1+1)) - set(rest_set_1) - train_set_1
rest_set_2_add = set(i for i in range(max_val1,max_val2+1)) - set(rest_set_2) - train_set_2

rest_set_1.extend(list(rest_set_1_add))#加入了dangling的test的kg1
rest_set_2.extend(list(rest_set_2_add))#加入了dangling的test的kg2
Unlabeled_num = len(rest_set_1 + rest_set_2)

rest_set_11 = rest_set_1
rest_set_22 = rest_set_2

data_set_1 = list(train_set_1) + rest_set_1
data_set_2 = list(train_set_2) + rest_set_2

print("len(data_set_1)")
print(len(data_set_1))
print("len(data_set_2)")
print(len(data_set_2))

def consolidated_test(tag):
    dir_path = fig_save
    name=''
    rest_set=None
    train_set=None
    if tag==1:
        name='extract_node_1'
        rest_set=rest_set_1
        train_set=train_set_1
    elif tag==2:
        name='extract_node_2'
        rest_set=rest_set_2
        train_set=train_set_2
    else:
        print("nonsense")
        return
    
    read_pd = pd.read_csv(os.path.join(dir_path, name), sep="\t", header=None, names=['source'])
    extract_node = set(read_pd['source'].values.tolist())
    extract_node = extract_node.intersection(train_set)#remove the training set,
    
    label1 = set(rest_set)
    label11 = list(extract_node.intersection(label1))
    
    precision = float(len(label11))/len(extract_node)
    recall = float(len(label11))/len(label1)
    true_e = []
    if tag==1:
        true_e = [e2 for e1, e2 in zip(rest_set_1, rest_set_2) if e1 in label11]
    elif tag==2:
        true_e = [e1 for e1, e2 in zip(rest_set_1, rest_set_2) if e1 in label11]
    
    candidate_list = true_e + list(set(rest_set)-set(true_e))#final to list
    Lvec, Rvec, _ = get_embedding(label11, candidate_list)
    
    consolidated_evaluater = evaluate(dev_pair,len(candidate_list))
    
    print("Lvec")
    print(len(Lvec))
    print("Rvec")
    print(len(Rvec))
    
    hits1, hits5, hits10, mrr = consolidated_evaluater.test_real(Lvec, Rvec)
    #two_step:
    Prec = hits1*precision
    Rec = hits1*recall
    F1 = 2 * Prec * Rec / (Prec + Rec)
    print("hits1",hits1,"Prec",Prec,"Rec",Rec,"F1",F1)

def save_embedding(index_a, index_b, train_pair, dev_pair):
    Lvec, Rvec, _ = get_embedding(index_a, index_b)
    result = np.concatenate((Lvec, Rvec))
    print("len result", len(result))
    embedding_name = 'embedding.txt'
    label_name = 'label.txt'
    embedding_path = os.path.join(fig_save, embedding_name)
    label_path = os.path.join(fig_save, label_name)
    node_size = result.shape[0]
    
    print("node_size")
    print(node_size)
    
    label = [[0]]*node_size
    pair = np.concatenate((train_pair, dev_pair))
    for i in pair:
        source_node=i[0]
        target_node=i[1]
        label[source_node]=[1]
        label[target_node]=[1]
    # label.astype(int)
    np.savetxt(label_path, label, '%d')
    np.savetxt(embedding_path, result)
    import sys
    sys.exit()
    return 

pi_p_u = train_len/(train_len + Unlabeled_num)
pi_p = (Unlabeled_num*train_len/(train_len + Unlabeled_num) + train_len)/(train_len + Unlabeled_num)

pi_p = pi_p_u = train_len/(train_len + Unlabeled_num)

pi_p = np.array(pi_p)
pi_p_u = np.array(pi_p_u)
# pi_p_u_copy = 0
print("pi_p, pi_p_u :", pi_p, pi_p_u)


def see_pi(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    print("train_len :", train_len)

    test_node = rest_set_1 + rest_set_2
    for i in test_node:
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FN+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TN+=1
            continue
    
    pos_num = TP + FP
    pi_p_u = pos_num/Unlabeled_num
    pi_p = (pos_num+train_len)/(train_len + Unlabeled_num)
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    test_node = train_list_1 + train_list_2 + rest_set_1 + rest_set_2
    for i in test_node:
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FN+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TN+=1
            continue
    
    # pos_num = TP + FP
    # pi_p = (pos_num)/(train_len + Unlabeled_num)
    
    print("pi_p, pi_p_u:", pi_p, pi_p_u)
    return pi_p, pi_p_u
    
    
def pi_generate(out_logits,y_true_n,pred_labels,y_labels,node_size):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # print("train_len :", train_len)
 
    test_node = rest_set_1 + rest_set_2
    for i in test_node:
        if pred_labels[i] == 0 and y_labels[i] == 0:
            TP+=1
            continue
        if pred_labels[i] == 0 and y_labels[i] == 1:
            FP+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 0:
            FN+=1
            continue
        if pred_labels[i] == 1 and y_labels[i] == 1:
            TN+=1
            continue
    
    pos_num = TP + FP
    pi_p_u = pos_num/Unlabeled_num
    pi_p = (train_len + pos_num)/(train_len + Unlabeled_num)
    
#     TP = 0
#     FP = 0
#     FN = 0
#     TN = 0
#     test_node = train_list_1 + train_list_2 + rest_set_1 + rest_set_2
#     for i in test_node:
#         if pred_labels[i] == 0 and y_labels[i] == 0:
#             TP+=1
#             continue
#         if pred_labels[i] == 0 and y_labels[i] == 1:
#             FP+=1
#             continue
#         if pred_labels[i] == 1 and y_labels[i] == 0:
#             FN+=1
#             continue
#         if pred_labels[i] == 1 and y_labels[i] == 1:
#             TN+=1
#             continue
    
#     pos_num = TP + FP
#     pi_p = (pos_num)/(train_len + Unlabeled_num)
    
    return np.array(pi_p), np.array(pi_p_u)

stop_train = False
loss_last = 0

def epoch_pi_generate(loss):
    global loss_last
    global pi_p
    global pi_p_u
    global stop_train

    bound = pi_p_u*3.0*1e-2
    loss_bound = 7*1e-4
    loss_bound = 1e-3
    # bound = pi_p*1e-2
    print("bound", bound)
    
    print("loss_bound", loss_bound)
    delta_loss = loss_last - loss
    if delta_loss < 0:
        delta_loss = - delta_loss
    print("delta_loss", delta_loss)
    if delta_loss < loss_bound:
        stop_train = True
        print("Stop loss") 
    
    _, _, out_logits = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
    y_true_n = np.array(y_true)
    pred_labels = Pred_logits(out_logits)
    y_labels = np.argmax(y_true_n, axis=1)
                    
    print("generate new pi")
    new_pi_p, new_pi_p_u = pi_generate(out_logits,y_true_n,pred_labels,y_labels,node_size)
    delta = pi_p_u - new_pi_p_u
    # delta = get_delta(out_logits,y_true_n,pred_labels,y_labels,node_size)
    if delta < 0:
        delta = - delta
    print("delta", delta)
    if delta < bound:
        stop_train = True
        print("Stop pi_p")
    pi_p = new_pi_p
    pi_p_u = new_pi_p_u
    print("pi_p:", pi_p)
    print("pi_p_u:", pi_p_u)
    
for I in range(1):
    for j in range(1):
        for k in range(1):
            for l in range(1):

                node_hidden = node_hidden_[I]
                rel_hidden = node_hidden
                batch_size = 2048  #
                batch_size = 1024  #
                batch_size = 768  #
                batch_size = 512  #
                # batch_size = 256  #
                dropout_rate = dropout_rate_[l]  #
                depth = depth_[j]
                ratio = ratio_[k]
                tar = int(node_size * ratio)
                
                model,get_emb = get_trgat(dropout_rate=dropout_rate,
                                          node_size=node_size,
                                          rel_size=rel_size,
                                          depth=depth,
                                          gamma =gamma,
                                          node_hidden=node_hidden,
                                          rel_hidden=rel_hidden,
                                          lr=lr,
                                          new_indexes=new_indexes)
                
                model.summary()
                
                evaluater = evaluate(dev_pair,u_size)#说明dev_pair和u_size的关系，
                #dev是同等维度的计算，这个只计算了origin
                #u_size是考虑所有的节点
                print("see u_size")
                print(u_size)#只有unlabeled
                
                np.random.shuffle(rest_set_1)
                np.random.shuffle(rest_set_2)
                
                train_pair_=train_pair
                # re_weight = []
                Pi_p = []
                Pi_p_u = []
                Loss = []
                epoch = 7
                flag = True

                for turn in range(6):#6 for GA16k
                    if stop_train==True:
                        break
                    for i in trange(epoch):
                        if stop_train==True:
                            break
                        np.random.shuffle(train_pair_)
                        for pairs in [train_pair_[i * batch_size:(i + 1) * batch_size] for i in
                                      range(len(train_pair_) // batch_size + 1)]:
                            if stop_train==True:
                                break
                            if len(pairs) == 0:
                                continue
                            # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs, y_true, indices_, un_indices]
                            if turn==1 and flag:
                                flag = False  
                                epoch_pi_generate(0)
                            print("train spyer:", pi_p, pi_p_u)
                            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs, y_true, indices_, un_indices, pi_p, pi_p_u]
                            inputs = [np.expand_dims(item, axis=0) for item in inputs]
                            align_list = pairs
                            model.train_on_batch(inputs, np.zeros((1, 1)))
                            loss = model.predict_on_batch(inputs)[0][0]
                            
                            print(loss)
                            
                            Loss.append(loss)
                            
                            _, _, out_logits = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
                            y_true_n = np.array(y_true)
                            pred_labels = Pred_logits(out_logits)
                            y_labels = np.argmax(y_true_n, axis=1)
                            
                            pi_p_0, pi_p_u_0 = see_pi(out_logits,y_true_n,pred_labels,y_labels,node_size)
                            Pi_p.append(pi_p_0)
                            Pi_p_u.append(pi_p_u_0)
                            if turn!=0:    
                                epoch_pi_generate(loss)
                            loss_last = loss
                            # epoch_pi_generate()
                            
                    # save_embedding(data_set_1, data_set_2, train_pair, dev_pair)
                    
                    print("test_dev_dev")
                    # out_logits = test_dev_dev()
                    print("out_logits")
                    _, _, out_logits = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
                    print("y_true_n")
                    y_true_n = np.array(y_true)
                    pred_labels = Pred_logits(out_logits)
                    y_labels = np.argmax(y_true_n, axis=1)

                    print("a")
                    classify_node_size(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("b")
                    classify_test_node(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("c")
                    precision1,recall1 = classify_test_node_1(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("cc")
                    precision1cc,recall1cc = classify_test_node_1_rec(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("d")
                    precision2,recall2 = classify_test_node_2(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("dd")
                    precision2dd,recall2dd = classify_test_node_2_rec(out_logits,y_true_n,pred_labels,y_labels,node_size)
                    print("e")
                    # test_dev_all(rest_set_1,rest_set_2,precision1,recall1,precision2,recall2)
                    # consolidated_test(1)#left to right and using precision and recall and F1
                    # consolidated_test(2)

                    print(I,j,k,l)
                    print("node_hidden",node_hidden_[I],"depth",depth_[j],"ratio",ratio_[k],"dropout_rate",dropout_rate_[l])
                    print("pi_p, pi_p_u",pi_p,pi_p_u)

#                     '''enhancement'''
#                     new_pair = []
#                     Lvec, Rvec, _ = get_embedding(rest_set_1, rest_set_2)
#                     A, B = evaluater.L_CSLS_cal(Lvec, Rvec)
                    
#                     print("rest_set_1") 
#                     print(len(rest_set_1))  
#                     print("rest_set_2") 
#                     print(len(rest_set_2)) 
                    
#                     for a, b in enumerate(A):
#                         if B[b] == a:
#                             new_pair.append([rest_set_11[b], rest_set_22[a]])
#                     train_pair_ = np.concatenate([train_pair_, np.array(new_pair)], axis=0)
#                     for e1, e2 in new_pair:
#                         if e1 in rest_set_11:
#                             rest_set_1.remove(e1)
                            
#                     for e1, e2 in new_pair:
#                         if e2 in rest_set_22:
#                             rest_set_2.remove(e2)   
#                     '''enhancement'''
                    epoch = 20
                
                def draw_visual():
                    #subgraph
                    layer=model.get_layer('e_encoder')
                    p = re_weight
                    att = np.argsort(p)[:10000]
                    set_att=set(np.unique(att+1))#抽出来的节点集合

                    #picture
                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    x = list(range(node_size))
                    red_index=[i[1] for i in pair]#目标图的上的目标点要标记为红色
                    green_index=[i[0] for i in pair]#原本图上的标记为绿色，看看绿色和红色，其他没用的标记为黑色
                    blue_index=[i for i in x if i not in red_index+green_index]#其他无用的点标记为蓝色

                    red_index=[i[1] for i in pair]#目标图的上的目标点要标记为红色
                    green_index=[i[0] for i in pair]#原本图上的标记为绿色，看看绿色和红色，其他没用的标记为黑色
                    blue_index=[i for i in x if i not in red_index+green_index]#其他无用的点标记为蓝色

                    colors = ['r' if i in red_index else 'g' if i in green_index else 'b' for i in x]
                    size = 5
                    print("xx")
                    print(np.shape(x))
                    print("yy")

                    y = layer.get_weights()[0].flatten()
                    print(np.shape(y))
                    plt.scatter(x, y, c=colors, s=size)
                    # 添加标题和坐标轴标签
                    plt.title("My Numpy Point Chart")
                    plt.xlabel("Node Number")
                    plt.ylabel("Reweighting")
                    # 显示图表
                    plt.show()
                    plt.savefig(fig_save+'3colors.png')
                    plt.clf()
                    print("3colors is upp")

                    #embedding的可视化
                    from sklearn.decomposition import PCA
                    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
                    inputs = [np.expand_dims(item, axis=0) for item in inputs]
                    vec, _= get_emb.predict_on_batch(inputs)#可视化准备

                    data = vec
                    cluster_colors = colors
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(data)
                    s = 2 #size

                    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_colors, s=s)
                    plt.title("My Numpy Point Chart")
                    plt.xlabel("X Label")
                    plt.ylabel("Y Label")
                    plt.show()
                    plt.savefig(fig_save+'embedding_for_classification_2D_1.png')
                    plt.clf()

                    #embedding的可视化
                    n_r = len(red_index)
                    n_g = len(green_index)
                    n_b = len(blue_index)
                    #R S
                    #related and source
                    d = np.concatenate((vec[red_index], vec[green_index]),axis=0)
                    c = ['r']*n_r + ['g']*n_g
                    reduced_data = pca.fit_transform(d)
                    s = 1

                    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=c, s=s)
                    plt.title("related and source")
                    plt.xlabel("X Label")
                    plt.ylabel("Y Label")
                    plt.show()
                    plt.savefig(fig_save+'R_S.png')
                    plt.clf()
                    #R U
                    #related and unrelated
                    d = np.concatenate((vec[red_index], vec[blue_index]),axis=0)
                    c = ['r']*n_r + ['b']*n_b
                    reduced_data = pca.fit_transform(d)
                    s = 1

                    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=c, s=s)
                    plt.title("related and unrelated")
                    plt.xlabel("X Label")
                    plt.ylabel("Y Label")
                    plt.show()
                    plt.savefig(fig_save+'R_U.png')
                    plt.clf()
                    #U S
                    #unrelated and source
                    d = np.concatenate((vec[blue_index], vec[green_index]),axis=0)
                    c = ['b']*n_b + ['g']*n_g
                    reduced_data = pca.fit_transform(d)
                    s = 1

                    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=c, s=s)
                    plt.title("unrelated and source")
                    plt.xlabel("X Label")
                    plt.ylabel("Y Label")
                    plt.show()
                    plt.savefig(fig_save+'U_S.png')
                    plt.clf()
                    
                    return
            
                def draw_pi(Pi_p, Pi_p_u):
                    # import seaborn as sns
                    import matplotlib.pyplot as plt
                    
                    pi_p_draw = Pi_p
                    pi_p_u_draw = Pi_p_u
                    
                    x_data = range(1, len(pi_p_draw)+1)
                    # y_data = [10, 15, 7, 10, 12]

                    # 绘制折线图
                    plt.plot(x_data, pi_p_draw, marker='o', linestyle='-', label='pi_p')
                    plt.plot(x_data, pi_p_u_draw, marker='s', linestyle='--', label='pi_p_u')
                    
                    pi_p_gt = (train_len + dev_len)/(train_len + Unlabeled_num)
                    pi_p_u_gt = dev_len/Unlabeled_num
                    plt.axhline(y=pi_p_gt, color='blue', linestyle='--')
                    plt.axhline(y=pi_p_u_gt, color='orange', linestyle='--')
                    plt.axvline(x=80, color='red', linestyle='--')
                    
                    plt.xlabel('epoch')
                    plt.ylabel('Ratio(%)')
                    plt.title('Ratio converges with epoch.')
                    plt.grid(True)  # 添加网格线
                    plt.savefig(fig_save+'Ratio_axhline.png')
                    plt.show()
                    np.savetxt(fig_save+'Pi_p_temp_1e.txt', pi_p_draw, fmt='%f', delimiter=',')
                    np.savetxt(fig_save+'Pi_p_u_temp_1e.txt', pi_p_u_draw, fmt='%f', delimiter=',')
                    
                def draw_loss(loss):
                    # import seaborn as sns
                    import matplotlib.pyplot as plt
                    print(type(loss))
                    print(loss)
                    x_data = range(1, len(loss)+1)
                    # y_data = [10, 15, 7, 10, 12]
                    plt.clf()
                    # 绘制折线图
                    plt.plot(x_data, loss, marker='o', linestyle='-', label='pi_p')
                    plt.axvline(x=80, color='red', linestyle='--')
                    
                    plt.xlabel('epoch')
                    plt.ylabel('Loss')
                    plt.title('Loss converges with epoch.')
                    plt.grid(True)  # 添加网格线
                    plt.savefig(fig_save+'Loss.png')
                    plt.show()
                    
                    np.savetxt(fig_save+'loss_1e.txt', loss, fmt='%f', delimiter=',')
                    
                # draw_pi(Pi_p, Pi_p_u)
                
                # draw_loss(Loss)
                
def save():
    #将有对应的节点提取出来
    pred_labels_list = []
    for i in range(len(pred_labels)):
        if pred_labels[i] == 0:
            pred_labels_list.append(i)

    pred_labels_list_1 = []
    pred_labels_list_2 = []

    for i in range(len(pred_labels)):
        if pred_labels[i] == 0:
            if i<=max_val1:
                pred_labels_list_1.append(i)
            else:
                pred_labels_list_2.append(i)

    import pandas as pd
    extract_node = pd.DataFrame(pred_labels_list, columns=['source'])
    extract_node.to_csv(os.path.join(fig_save, 'extract'), sep="\t", header=None, index=None)
    n_pred = len(pred_labels_list)#提取出来的节点数量
    print("n_pred")
    print(n_pred)

    extract_node_1 = pd.DataFrame(pred_labels_list_1, columns=['source'])
    extract_node_2 = pd.DataFrame(pred_labels_list_2, columns=['source'])
    extract_node_1.to_csv(os.path.join(fig_save, 'extract_node_1'), sep="\t", header=None, index=None)
    extract_node_2.to_csv(os.path.join(fig_save, 'extract_node_2'), sep="\t", header=None, index=None)

    n_pred_1 = len(pred_labels_list_1)#提取出来的节点数量
    print("n_pred_1")
    print(n_pred_1)

    n_pred_2 = len(pred_labels_list_2)#提取出来的节点数量
    print("n_pred_2")
    print(n_pred_2)
    return 
    '''分类模型'''

# save()

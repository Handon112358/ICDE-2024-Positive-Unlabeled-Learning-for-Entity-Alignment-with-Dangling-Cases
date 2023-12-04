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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true = load_data("data/16k_update/",train_ratio=0.30)
# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true = load_data("data/subgraph/ja_sub/",train_ratio=0.30)
# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true = load_data("data/subgraph/zh_sub/",train_ratio=0.30)
# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features,target_dis_tag,new_indexes,y_true = load_data("data/subgraph/fr_sub/",train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

node_size = adj_features.shape[0]#节点数目
rel_size = rel_features.shape[1]#关系数目
triple_size = len(adj_matrix)#三元组数目
node_hidden = 128
rel_hidden = 128
batch_size = 1024#
dropout_rate = 0.3#
lr = 0.005#
gamma = 1#
depth = 2#
tar=int(node_size*0.75)

indices_ = np.concatenate((train_pair[:,0], train_pair[:,1]), axis=0)
indices_ = np.concatenate((indices_, dev_pair[:,0]), axis=0)
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
tar_tensor=tf.tensor_scatter_nd_update(tar_tensor, indices, updates)#目标的向量

def get_embedding(index_a, index_b, vec=None):
    if vec is None:
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        vec, re_weight, out_logits = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec, re_weight, out_logits

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


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
    [e_feature,p_e,d_e] = e_encoder([ent_feature] + opt)#p_e注意力分布
    [r_feature,p_r,d_r] = r_encoder([rel_feature] + opt)

    def reweight_div(tensor):#返回注意力分布
        p_e,p_r=tensor
        return (p_e + p_r)/2
        
    final_reweight=Lambda(reweight_div)([p_e, p_r])
  
    e_out_feature = Concatenate(-1)([e_feature, d_e])#先进行e
    r_out_feature = Concatenate(-1)([e_feature, d_e])#再进行r
    out_feature = Concatenate(-1)([e_out_feature, r_out_feature])#最后合并
    
    out_feature = Dropout(dropout_rate)(out_feature)
    out_logits = Dense(2,activation='softmax')(out_feature)
    
    alignment_input = Input(shape=(None, 2))#6
    y_true = Input(shape=(None,2))#7
    
    indices = Input(shape=(None,), dtype='int32')#8
    un_indices = Input(shape=(None,),dtype='int32')#9
    
    def data_y_p_p_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        # y_train = K.gather(y_true, indices)
        n = p_size
        y_train = tf.constant([[1.0, 0.0]] * n)
        return y_train

    def data_y_p_n_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        # y_train = K.gather(y_true, indices)
        n = p_size
        y_train = tf.constant([[0.0, 1.0]] * n)
        return y_train
    
    def data_y_u_n_label(tensor):
        y_true = tensor[0]
        indices = tensor[1]
        # y_train = K.gather(y_true, indices)
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
        #anchor len(train_pair)
        #train_pair
        # pi_p = 2.0 * tf.cast(K.shape(y_p_p)[0], tf.float32) / (tf.cast(K.shape(y_p_p)[0], tf.float32) + tf.cast(K.shape(y_u_n)[0], tf.float32))
        
        # pi_p_p = tf.cast(K.shape(y_p_p)[0], tf.float32) / (tf.cast(K.shape(y_u_n)[0], tf.float32))
        # pi_p_n = (tf.cast(K.shape(y_p_p)[0], tf.float32) - tf.cast(2.0*anchor_size, tf.float32))/ (tf.cast(K.shape(y_p_p)[0], tf.float32) + tf.cast(K.shape(y_u_n)[0], tf.float32))
        beta = 1.0
        pi_p = beta*tf.cast(K.shape(y_p_p)[0], tf.float32) / (tf.cast(K.shape(y_u_n)[0], tf.float32))
        
        CE_nn = K.relu(CE_u_n - pi_p*CE_p_n)#max
        gama = 0.5
        return gama*tensor[0] + (1 - gama) * (pi_p * CE_p_p + CE_nn)

    loss = Lambda(Loss)([loss_EA, y_p_p, y_p_n, y_u_n, logits_y_p, logits_y_u])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input, y_true, indices, un_indices], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))
    
    feature_model = keras.Model(inputs=inputs, outputs=[out_feature,final_reweight,out_logits])
    return train_model, feature_model

node_hidden_=[128]#4
depth_ = [2]#2
#ratio_=[0.75,0.80,0.85,0.9]#5
ratio_=[0.75]
#dropout_rate_=[0.2,0.3,0.4,0.5]#4
dropout_rate_=[0.5]
flag=1

pair = dev_pair.tolist() + train_pair.tolist()

for I in range(1):
    for j in range(1):
        for k in range(1):
            for l in range(1):

                node_hidden = node_hidden_[I]
                rel_hidden = node_hidden
                batch_size = 1024  #
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

                evaluater = evaluate(dev_pair)

                rest_set_1 = [e1 for e1, e2 in dev_pair]
                rest_set_2 = [e2 for e1, e2 in dev_pair]
                np.random.shuffle(rest_set_1)
                np.random.shuffle(rest_set_2)
                
                train_pair_=train_pair
                
                epoch = 20
                for turn in range(5):
                    for i in trange(epoch):
                        np.random.shuffle(train_pair_)
                        for pairs in [train_pair_[i * batch_size:(i + 1) * batch_size] for i in
                                      range(len(train_pair_) // batch_size + 1)]:
                            if len(pairs) == 0:
                                continue
                            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs, y_true, indices_, un_indices]
                            inputs = [np.expand_dims(item, axis=0) for item in inputs]
                            align_list = pairs
                            model.train_on_batch(inputs, np.zeros((1, 1)))

                    Lvec, Rvec, _ , _ = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
                    evaluater.test(Lvec, Rvec)
                    epoch = 5
                
                #AP
                Lvec, Rvec, re_weight, out_logits = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
                
                evaluater.test(Lvec, Rvec)
                print(I,j,k,l)
                print("node_hidden",node_hidden_[I],"depth",depth_[j],"ratio",ratio_[k],"dropout_rate",dropout_rate_[l])
                print("Next")
                
                #subgraph
                layer=model.get_layer('e_encoder')
                # p=layer.get_weights()[0].flatten()
                p = re_weight
                att = np.argsort(-p)[:13000]
                set_att=set(np.unique(att+1))#抽出来的节点集合
                
                #eval
                ans=[i[1] for i in train_pair]+[i[1] for i in dev_pair]
                ans_s=ans+[i[0] for i in train_pair]+[i[0] for i in dev_pair]
                set_ans=set(np.unique(ans))
                set_ans_s=set(np.unique(ans_s))
                
                S3 = set_att.intersection(set_ans)
                print("S3_len:")
                print(len(S3))
                S4 = set_att.intersection(set_ans_s)
                print("S4_len:")
                print(len(S4))
                
                y_true_n = np.array(y_true)
                print("y_true_n")
                print(y_true_n)
                pred_labels = np.argmax(out_logits, axis=1)
                print("pred_labels")
                print(pred_labels)
                y_labels = np.argmax(y_true_n, axis=1)
                print("y_labels")
                print(y_labels)
                # # TP
                # TP = np.sum(pred_labels == 1  & y_labels == 1)
                # # FP   
                # FP = np.sum(pred_labels == 1 & y_labels == 0)  
                # # FN       
                # FN = np.sum(pred_labels == 0 & y_labels == 1)
                # # TN: 真实为负例且正确预测为负例的数目
                # TN = np.sum(pred_labels == 0 & y_labels == 0)
                
                TP = 0
                FP = 0
                FN = 0
                TN = 0
                for i in range(node_size):
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
                
                print("TP")
                print(TP)
                print("FP")
                print(FP)
                print("FN")
                print(FN)
                print("TN")
                print(TN)
                
                Accuracy = (TP + TN) / (TP + FP + TN + FN)
                precision = TP / (TP + FP)
                recall = TP /(TP + FN)
                F1 = 2 * precision * recall / (precision + recall)
                print("test and train")
                print("Accuracy")
                print(Accuracy)
                print("precision")
                print(precision)
                print("recall")
                print(recall)
                print("F1")
                print(F1)
                print("--------------------------")
                
                TP = 0
                FP = 0
                FN = 0
                TN = 0
                train_node=[i[0] for i in pair]#原本图上的标记为绿色，看看绿色和红色，其他没用的标记为黑色
                for i in range(node_size):
                    if i in train_node:
                        continue
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
                Accuracy = (TP + TN) / (TP + FP + TN + FN)
                precision = TP / (TP + FP)
                recall = TP /(TP + FN)
                F1 = 2 * precision * recall / (precision + recall)
                print("test")
                print("Accuracy")
                print(Accuracy)
                print("precision")
                print(precision)
                print("recall")
                print(recall)
                print("F1")
                print(F1)
                
                
                #picture
                import seaborn as sns
                import matplotlib.pyplot as plt
                data=re_weight

                threshold=0.5
                count = np.sum(data > threshold)
                print("大于 threshold 的元素个数：", count)
                
                # fig_save='data/subgraph/zh_sub/'
                fig_save='data/16k_update/'
                # x, y
                x = list(range(node_size))
                y = data
                #color
                red_index=[i[1] for i in pair]#目标图的上的目标点要标记为红色
                green_index=[i[0] for i in pair]#原本图上的标记为绿色，看看绿色和红色，其他没用的标记为黑色
                blue_index=[i for i in x if i not in red_index+green_index]#其他无用的点标记为蓝色
                
                val_red=data[red_index]
                val_green=data[green_index]
                val_blue=data[blue_index]
                
                sns.distplot(val_red, hist=False, label='Related Node')
                sns.distplot(val_green, hist=False, label='Source Node')
                sns.distplot(val_blue, hist=False, label='Unrelated Node')
                
                plt.legend()   # 图例
                plt.xlabel('Value')
                plt.ylabel('Probability')
                plt.savefig(fig_save+'3colors_dis_after_W.png')
                plt.clf()
                print("3colors_dis is down")
                
                colors = ['r' if i in red_index else 'g' if i in green_index else 'b' for i in x]
                size = 5
                print("xx")
                print(np.shape(x))
                print("yy")
                print(np.shape(y))
                y = layer.get_weights()[0].flatten()
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
                vec, _, _ = get_emb.predict_on_batch(inputs)#可视化准备
                
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

                #weight
                layer=model.get_layer('e_encoder')
                weight=layer.weights
                e_encoder=weight[0]
                name=weight[0].name

                print("weight")
                print(weight)
                print("name")
                print(name)
                print("weight__:")
                data=layer.get_weights()[0].flatten()
                print(data)

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
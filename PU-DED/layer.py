from __future__ import absolute_import

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

tar_tensor=tf.zeros([2], dtype=tf.float32)

class TarRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, tar, factor=1.0):
        self.factor = factor
        self.tar = tar

    def __call__(self, x):
        #return 10000*(tf.reduce_sum(tf.sigmoid(x))-self.tar)+10000*tf.reduce_sum(tf.square(tf.multiply(tf.subtract(tf.sigmoid(x),tar_tensor),tar_tensor)))
        # return 10000*tf.reduce_sum(x-self.tar)+10000*tf.reduce_sum(tf.square(tf.multiply(tf.subtract(x,tar_tensor),tar_tensor)))
        # return 10000*(tf.reduce_sum(tf.tanh(x))-self.tar)+10000*tf.reduce_sum(tf.square(tf.multiply(tf.subtract(tf.tanh(x),tar_tensor),tar_tensor)))
        # return 10000*(tf.reduce_sum(tf.tanh(x))-self.tar)
        return 10000*tf.reduce_sum(tf.square(tf.multiply(tf.subtract(tf.tanh(x),tar_tensor),tar_tensor)))

class OrthRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, x):
        return self.factor*K.sum(K.square(tf.matmul(x, x, transpose_a=True) - tf.eye(tf.shape(x)[0])))
        # return 0
    
class DNW_layer(Layer):
    def __init__(self,
                 node_size,
                 tar,
                 tar_tensor,
                 **kwargs):
        self.node_size = node_size
        self.tar = tar
        self.tar_tensor = tar_tensor
        self.dynamic_node_kernel = []
        super(DNW_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        dynamic_kernel = self.add_weight(shape=(self.node_size, 1),
                                         initializer=initializers.Constant(value=1),
                                         regularizer=TarRegularizer(self.tar),
                                         constraint=None,
                                         name='dynamic_node_kernel')
        self.dynamic_node_kernel.append(dynamic_kernel)

    def call(self, inputs):
        return tf.multiply(inputs[0],tf.sigmoid(self.dynamic_node_kernel[0]))


class NR_GraphAttention(Layer):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 tar,
                 tar_tensor,
                 new_indexes,
                 depth = 1,
                 W_orth = True,
                 use_p = True,
                 use_w = False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 att_dis=False,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth
        self.W_orth = W_orth
        self.use_p = use_p
        self.tar = tar
        self.tar_tensor = tar_tensor
        self.new_indexes = new_indexes[0]#2
        self.new_indices = new_indexes[1]#0
        if self.tar >= node_size:
            self.tar = node_size

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False
        
        self.dynamic_node_kernel = []
        self.att_reweight = []
        self.biases = []        
        self.attn_kernels = []  
        self.gat_kernels = []
        self.gate_kernels = []
        self.w_key = []#关系的正交变换
        self.att_dis = att_dis
        # self.w_class = []
        
        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        rel_F = input_shape[1][-1]
        self.ent_F = node_F
        ent_F = self.ent_F
        
        if self.use_p:
            # for l in range(1):
                # dynamic_kernel = self.add_weight(shape=(self.node_size,1),
                #                                 initializer=initializers.Constant(value=1),
                #                                 regularizer=TarRegularizer(self.tar),
                #                                 constraint=self.kernel_constraint,
                #                                 name='dynamic_node_kernel_{}'.format(l))
            dynamic_kernel = self.add_weight(shape=(self.node_size,1),
                                            initializer=initializers.Constant(value=1),
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint,
                                             name='dynamic_node_kernel')
                                            # name='dynamic_node_kernel_{}'.format(l))
            # self.dynamic_node_kernel.append(dynamic_kernel)
            self.dynamic_node_kernel = dynamic_kernel

        if self.W_orth:
            for l in range(self.depth):
                # w_key = self.add_weight(shape=(ent_F,ent_F),
                #                                 initializer=self.kernel_initializer,
                #                                 regularizer=OrthRegularizer(factor=1.0),
                #                                 constraint=self.kernel_constraint,
                #                                 name='trans_w_{}'.format(l))
                w_key = self.add_weight(shape=(ent_F,ent_F),
                                                initializer=self.kernel_initializer,
                                                regularizer=OrthRegularizer(factor=1.0),
                                                constraint=self.kernel_constraint,
                                                name='trans_w_{}'.format(l))
                self.w_key.append(w_key)
        
        self.gate_kernel = self.add_weight(shape=(ent_F*(self.depth+1),ent_F*(self.depth+1)),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='gate_kernel')

        # self.proxy = self.add_weight(shape=(64, node_F*(self.depth+1)),
        #                            initializer=self.attn_kernel_initializer,
        #                            regularizer=self.attn_kernel_regularizer,
        #                            constraint=self.attn_kernel_constraint,
        #                            name='proxy')
        self.proxy = self.add_weight(shape=(128, node_F*(self.depth+1)),
                                   initializer=self.attn_kernel_initializer,
                                   regularizer=self.attn_kernel_regularizer,
                                   constraint=self.attn_kernel_constraint,
                                   name='proxy')
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(1, ent_F*(self.depth+1)),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   name='bias')
            
        for l in range(self.depth):
            self.attn_kernels.append([])
            for head in range(self.attn_heads):                
                attn_kernel = self.add_weight(shape=(1*node_F ,1),
                                       initializer=self.attn_kernel_initializer,
                                       regularizer=self.attn_kernel_regularizer,
                                       constraint=self.attn_kernel_constraint,
                                       name='attn_kernel_self_{}'.format(head))
                self.attn_kernels[l].append(attn_kernel)
                
        self.built = True

    def call(self, inputs):
        # print("call")
        outputs = []
        features = inputs[0]
        if self.use_p:
            features = tf.multiply(inputs[0],tf.tanh(self.dynamic_node_kernel[0]))
        
        rel_emb = inputs[1]
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2],axis = 0),dtype = "int64"),
                         K.ones_like(inputs[2][0,:,0]),(self.node_size,self.node_size))

        sparse_indices = tf.squeeze(inputs[3],axis = 0)  
        sparse_val = tf.squeeze(inputs[4],axis = 0)

        features = self.activation(features)
        outputs.append(features)
                        
        for l in range(self.depth):
            features_list = []
            dynamic_kernel = None
            tf_dynamic_kernel = None
            if self.use_p:
                # dynamic_kernel = self.dynamic_node_kernel[0]
                dynamic_kernel = self.dynamic_node_kernel
                tf_dynamic_kernel = tf.tanh(dynamic_kernel)
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]  
                rels_sum = tf.SparseTensor(indices=sparse_indices,values=sparse_val,dense_shape=(self.triple_size,self.rel_size))
                rels_sum = tf.sparse_tensor_dense_matmul(rels_sum,rel_emb)
                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                
                if self.W_orth:
                    w_key = self.w_key[l]
                    rels_sum_source = K.gather(rels_sum, self.new_indices)
                    rels_sum_source = K.dot(rels_sum_source, w_key)
                    rels_sum = tf.tensor_scatter_nd_update(rels_sum, self.new_indexes, rels_sum_source)
                
                neighs = K.gather(features,adj.indices[:,1])
                
                if self.use_p:
                    att_neighs = K.gather(tf_dynamic_kernel, adj.indices[:, 1])
                    rels_sum = tf.multiply(rels_sum, att_neighs)
                
                neighs = neighs - 2 * tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum

                att = K.squeeze(K.dot(rels_sum,attention_kernel),axis = -1) #K.concatenate([selfs,neighs,rels_sum])
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.sparse_softmax(att)
                
                if self.att_dis:
                    att_dense = tf.sparse_tensor_to_dense(att)
                    att_col_counts = tf.cast(tf.reduce_sum(tf.cast(att_dense > 0 , tf.float32), 0), tf.float32)
                    att_col_sums = tf.reduce_sum(att_dense, 0)

                    self.att_reweight.append(tf.divide(att_col_sums, att_col_counts))

                new_features = tf.segment_sum (neighs*K.expand_dims(att.values,axis = -1),adj.indices[:,0])
                if self.use_p:
                    new_features = tf.multiply(new_features, tf_dynamic_kernel)
                features_list.append(new_features)
                

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)  # (N x KF')
            else:
                features = K.mean(K.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)
        
        outputs = K.concatenate(outputs)
        proxy_att = K.dot(tf.nn.l2_normalize(outputs,axis=-1),K.transpose(tf.nn.l2_normalize(self.proxy,axis=-1)))
        proxy_att = K.softmax(proxy_att,axis = -1)
        proxy_feature = outputs - K.dot(proxy_att,self.proxy)

        if self.use_bias:
            gate_rate = K.sigmoid(K.dot(proxy_feature,self.gate_kernel) + self.bias)
        else:
            gate_rate = K.sigmoid(K.dot(proxy_feature,self.gate_kernel))
        
        outputs = (gate_rate) * outputs + (1-gate_rate) * proxy_feature
        
        # if len(self.att_reweight)!=0:
        #     re_weight=tf.reduce_mean(self.att_reweight,axis=0)
        # else re_weight=tf.zeros(shape=(self.node_size,))
        node_p = self.dynamic_node_kernel

        if self.use_p and self.att_dis:
            re_weight=tf.reduce_mean(self.att_reweight,axis=0)
            return [outputs, re_weight, node_p]
        if self.att_dis:
            re_weight=tf.reduce_mean(self.att_reweight,axis=0)
            return [outputs, re_weight]
        if self.use_p:
            return [outputs, node_p]
        return [outputs]

    def compute_output_shape(self, input_shape):    
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth+1)
        re_weight_shape = self.node_size,
        p_shape = self.node_size,1
        if self.use_p and self.att_dis:
            return [node_shape, re_weight_shape, p_shape]
        if self.att_dis:
            return [node_shape, re_weight_shape]
        if self.use_p:
            return [node_shape, p_shape]
        return [node_shape]

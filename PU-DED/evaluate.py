import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *

class evaluate:
    def __init__(self,dev_pair,length):
        self.dev_pair = dev_pair
        self.length = length
        
        '''model 1'''
        Matrix_A = Input(shape=(None,None))
        Matrix_B = Input(shape=(None,None))
        def dot(tensor):
            k = 10
            A,B = [K.squeeze(matrix,axis=0) for matrix in tensor]
            A_sim = K.dot(A,K.transpose(B))
            return K.expand_dims(A_sim,axis=0)

        results = Lambda(dot)([Matrix_A,Matrix_B])
        self.sim_model = keras.Model(inputs = [Matrix_A,Matrix_B],outputs = results)

        '''model 2'''
        k = 10
        Matrix_A = Input(shape=(None,None))
        results = Lambda(lambda x: K.expand_dims(K.sum(tf.nn.top_k(K.squeeze(x,axis=0),k=k)[0],axis=-1) / k,axis=0))(Matrix_A)
        self.avg_model = keras.Model(inputs = [Matrix_A],outputs = results)

        '''model 3'''
        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=( None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,_ = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL = [K.expand_dims(m,axis=1) for m in [LR,RL]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            return K.expand_dims(rank[:,0],axis=0)

        rank = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.rank_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = rank)

        '''model 4'''
        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=( None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,_ = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL = [K.expand_dims(m,axis=1) for m in [LR,RL]]
            sim = 2*sim - K.transpose(RL)
            sim = sim - LR
            rank = tf.argsort(-sim,axis=-1)
            return K.expand_dims(rank[:,0],axis=0)

        rank = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.L_rank_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = rank)
        
        '''model 5'''
        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=(None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,ans_rank = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL,ans_rank = [K.expand_dims(m,axis=1) for m in [LR,RL,ans_rank]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,len(self.dev_pair)]),dtype="int32")))
            return K.expand_dims(results,axis=0)

        results = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.CSLS_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = results)

        '''model 6'''
        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=(None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,ans_rank = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL,ans_rank = [K.expand_dims(m,axis=1) for m in [LR,RL,ans_rank]]
            sim = 2*sim - LR
            sim = sim - K.transpose(RL)
            rank = tf.argsort(-sim,axis=-1)
            results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,self.length]),dtype="int32")))
            return K.expand_dims(results,axis=0)

        results = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.L_CSLS_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = results)
        
        '''model 7'''
        #Matrix_A = Input(shape=(None,None))
        #Ans_input = Input(shape=(None,))
        #def CSLS(tensor):
        #    sim,ans_rank = [K.squeeze(m,axis=0) for m in tensor]
        #    ans_rank = K.expand_dims(ans_rank,axis=1)
        #    rank = tf.argsort(-sim,axis=-1)
        #    results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,self.length]),dtype="int32")))
        #    return K.expand_dims(results,axis=0)

        #results = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        #self.R_CSLS_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = results)

    def CSLS_cal(self, Lvec,Rvec,evaluate = True,batch_size = 1024):
        L_sim,R_sim = [],[]
        # print("aaaaaaa")
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))

        LR,RL = [],[]
        # print("bbbbbbbb")
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
            RL.append(self.avg_model.predict([R_sim[epoch]]))

        if evaluate:
            results = []
            # print("ccccccc")
            for epoch in range(len(Lvec) // batch_size + 1):
                print(epoch)
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                result = self.CSLS_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)])
                results.append(result)
            # print("ddddddd")
            return np.concatenate(results,axis=1)[0]
        else:
            l_rank,r_rank = [],[]
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                r_rank.append(self.rank_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)]))
                l_rank.append(self.rank_model.predict([L_sim[epoch],np.concatenate(RL,axis=1),LR[epoch],np.expand_dims(ans_rank,axis=0)]))

            return np.concatenate(r_rank,axis=1)[0],np.concatenate(l_rank,axis=1)[0] 

    def L_CSLS_cal(self, Lvec,Rvec):
        L_sim,R_sim = [],[]
        batch_size=1024
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
        for epoch in range(len(Rvec) // batch_size + 1):    
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))

        LR,RL = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
        for epoch in range(len(Rvec) // batch_size + 1):
            RL.append(self.avg_model.predict([R_sim[epoch]]))

        l_rank,r_rank = [],[]
        for epoch in range(len(Rvec) // batch_size + 1):
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
            r_rank.append(self.rank_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)]))
        for epoch in range(len(Lvec) // batch_size + 1):
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Rvec)))])
            l_rank.append(self.rank_model.predict([L_sim[epoch],np.concatenate(RL,axis=1),LR[epoch],np.expand_dims(ans_rank,axis=0)]))

        return np.concatenate(r_rank,axis=1)[0],np.concatenate(l_rank,axis=1)[0]
    
    def test(self, Lvec,Rvec):
        results  = self.CSLS_cal(Lvec,Rvec)
        print("results")
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return hits1/len(Lvec),hits5/len(Lvec),hits10/len(Lvec),mrr/len(Lvec)
    
    def test_real(self, Lvec,Rvec):
        batch_size = 1024
        L_sim,R_sim = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
        for epoch in range(len(Rvec) // batch_size + 1):
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))

        LR,RL = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
        for epoch in range(len(Rvec) // batch_size + 1):
            RL.append(self.avg_model.predict([R_sim[epoch]]))      
        
        results = []
        for epoch in range(len(Lvec) // batch_size + 1):
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
            result = self.L_CSLS_model.predict([L_sim[epoch],LR[epoch],np.concatenate(RL,axis=1),np.expand_dims(ans_rank,axis=0)])
            results.append(result)
        results = np.concatenate(results,axis=1)[0]
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return hits1/len(Lvec),hits5/len(Lvec),hits10/len(Lvec),mrr/len(Lvec)
        
    def test_extract(self, Lvec, Rvec, label, dev):
        batch_size = 1024
        L_sim,R_sim = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
        for epoch in range(len(Rvec) // batch_size + 1):
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))

        LR,RL = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
        for epoch in range(len(Rvec) // batch_size + 1):
            RL.append(self.avg_model.predict([R_sim[epoch]]))      
        
        results = []
        for epoch in range(len(Lvec) // batch_size + 1):
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
            result = self.L_CSLS_model.predict([L_sim[epoch],LR[epoch],np.concatenate(RL,axis=1),np.expand_dims(ans_rank,axis=0)])
            results.append(result)
        results = np.concatenate(results,axis=1)[0]
        def cal(results,label):
            hits1,hits5,hits10,mrr = 0,0,0,0
            length = len(results[:,1])
            find = results[:,1]
            for x in range(length):
                y = find[x]
                if label[dev[x]]:
                    continue
                if y < 1:
                    hits1 += 1
                if y < 5:
                    hits5 += 1
                if y < 10:
                    hits10 += 1
                mrr += 1/(y + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results,label)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return 
        
        
    def test_real(self, Lvec,Rvec):
        batch_size = 1024
        results = []
        # L_sim,R_sim = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim = self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)])
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
            result = self.R_CSLS_model.predict([L_sim,np.expand_dims(ans_rank,axis=0)])
            results.append(result)
    
        results = np.concatenate(results,axis=1)[0]
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return hits1/len(Lvec),hits5/len(Lvec),hits10/len(Lvec),mrr/len(Lvec)

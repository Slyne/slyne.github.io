#-*- coding:utf-8 -*-
import numpy as np
import os

class Lda(object):
    def __init__(self, K, V, alpha, beta):
        self.K = K
        self.V = V
        self.alpha = alpha
        self.beta = beta
        assert len(alpha) == K
        assert len(beta) == V
    
    def learn(self, passages, iterations = 100):
        """
        passages: a list of passage, one passage is a list of words (ids)
        vocabulary: a list of vocabulary (ids)
        """
        # initialize
        M = len(passages)
        V = self.V
    
        self.num_mk = np.zeros((M, self.K))
        self.num_m = np.zeros(M)
        self.num_kv = np.zeros((self.K, V))
        self.num_k = np.zeros(self.K)
        self.topics = []
        for m, passage_m in enumerate(passages):
            cur_topic = []
            for n, word_mn in enumerate(passage_m):
                z_mn = np.random.choice(a=range(self.K), p=[1.0/self.K]*self.K)
                self.num_mk[m][z_mn] += 1
                self.num_m[m] += 1
                self.num_kv[z_mn][word_mn] += 1
                self.num_k[z_mn] += 1
                cur_topic.append(z_mn)
            self.topics.append(cur_topic)
        
        while iterations > 0:
            for m, passage_m in enumerate(passages):
                for n, word_mn in enumerate(passage_m):
                    v = word_mn
                    k = self.topics[m][n]
                    self.num_mk[m][k] -= 1
                    self.num_m[m] -= 1
                    self.num_kv[k][v] -= 1
                    self.num_k[k] -= 1
                    k_plus = self.sample(v, m)
                    self.num_mk[m][k_plus] += 1
                    self.num_m[m] += 1
                    self.num_kv[k_plus][v] += 1
                    self.num_k[k_plus] += 1
                    self.topics[m][n] = k_plus
            iterations -= 1
            if iterations % 5 == 0: 
                print(iterations)
        
        # calculate theta
        temp1 = np.asarray(self.num_mk, dtype=np.float32) + np.asarray(self.alpha)
        self.theta = temp1 / np.sum(temp1, axis=1)[:, np.newaxis]

        # calculate varphi
        temp1 = np.asarray(self.num_kv, dtype=np.float32) + np.asarray(self.beta)
        self.varphi = temp1 / np.sum(temp1, axis=1)[:, np.newaxis]
    
    def sample(self, v, m):
        p = []
        #normalization_alpha = 0  # alpha可以不用计算，相同的分母项
        for k in range(self.K):
            cur_prob = (self.num_kv[k][v] + self.beta[v]) * (self.num_mk[m][k]+self.alpha[k])
            normalization_beta = np.sum(self.num_kv[k] + self.beta) # 每个话题k下面的word的正则项都不同
            p.append(float(cur_prob)/normalization_beta)  
            #normalization_alpha += self.num_mk[m][k] + self.alpha[k]
        #p = np.array(p) / float(normalization_alpha))

        p = np.array(p, dtype=np.float32)
        normalization = sum(p)
        p = p/normalization
        try:
            k_plus = np.random.choice(a=range(self.K), p = p)
        except:
            print(p)
            
        return k_plus

import re
def get_passage(raw_passage):
    passage = []
    for idx, sentence in enumerate(raw_passage.split("\n")):
        if idx % 2 == 0:
            passage.append(sentence)
    return " ".join(passage)


'''
import nltk
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
'''

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = []
passage_names = []
for filename in os.listdir("TED"):
    passage_names.append(filename)
    print(filename)
    passage = get_passage(open(os.path.join("TED",filename), "r").read().lower())
    corpus.append(passage)

# 我们选择TED 2019的音频文本作为需要抽取话题的文本
# 下面的操作基本流程就是抽取出文本中tfidf值比较高的词来表示这个文本
# 最后所有文本用其单词的id的list来表示
vectorizer=TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(corpus)
vocabulary = [(idx, word) for  word,idx in vectorizer.vocabulary_.items()] # word->id
vocabulary = dict(vocabulary)
assert len(vocabulary) == len(tfidf.toarray()[0])
passage_ids = []
for passage_id, tfidf_array in enumerate(tfidf.toarray()):
    id_tfidf = sorted(enumerate(tfidf_array), key=lambda k:k[1], reverse=True)
    # only choose top 20 as keywords for this passage
    important_ids = [idx for idx, value in id_tfidf[0:50]]
    passage_id = [vectorizer.vocabulary_[word] \
                    for word in corpus[passage_id].split() \
                        if word in vectorizer.vocabulary_ and vectorizer.vocabulary_[word] in important_ids]

    passage_ids.append(passage_id)




print([(vocabulary[idx],idx) for idx in passage_ids[0]])

K = 10
alpha = np.ones(K)
V = len(vocabulary)
beta = np.ones(V)
lda = Lda(K, V, alpha, beta)
lda.learn(passage_ids,100)

np.save("theta", lda.theta)
np.save("varphi", lda.varphi)


print(passage_names[0])
theta0  = lda.theta[0]

print("passage topic distribute", theta0)
#print("topic word distribution", lda.varphi[topic])
most_similar = 0
min_distance = float("inf")

for idx in range(1,len(theta0)):
    distance = np.sum((lda.theta[idx]-theta0)**2)
    if distance < min_distance:
        min_distance = distance
        most_similar = idx

print(lda.theta[most_similar])
print(passage_names[most_similar])
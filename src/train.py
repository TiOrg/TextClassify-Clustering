#coding:utf-8 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import os
# from gensim import corpora, models, similarities, matutils

rootpath=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))           
datapath = os.path.join(os.path.sep, rootpath, 'data') 

# from sklearn import preprocessing
# from sklearn import metrics
# import matplotlib.pyplot as plt
print 'reading data...'
x=pickle.load(open(os.path.join(os.path.sep, datapath, 'tfidf_data.pkl'),'rb'))
y=pickle.load(open(os.path.join(os.path.sep, datapath, 'train_label.pkl'),'rb'))

x=np.array(x)
y=np.array(y)


def getCossim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*(np.linalg.norm(vec2)))


def getMaxSimilarity(topics, vector):
    maxValue = 0
    maxIndex = -1
    for k,cluster in topics.iteritems():
        oneSimilarity = np.mean([getCossim(vector, v) for v in cluster])
        if oneSimilarity > maxValue:
            maxValue = oneSimilarity
            maxIndex = k
    return maxIndex, maxValue



def single_pass(vectors, thres):
    topics = []
    dictCluster = {}
    numCluster = 0 
    cnt = 0
    topic = 0
    for vector in vectors: 
        if numCluster == 0:
            dictCluster[numCluster] = []
            dictCluster[numCluster].append(vector)

            numCluster += 1
    
        else:
            maxIndex, maxValue = getMaxSimilarity(dictCluster, vector)
            # print maxValue
            
            #join the most similar topic
            if maxValue > thres:
                dictCluster[maxIndex].append(vector)
                topic = maxIndex
            #else create the new topic
            else:
                dictCluster[numCluster] = []
                dictCluster[numCluster].append(vector)
                topic = numCluster
                numCluster += 1
        topics.append(topic)
        cnt += 1
        if cnt % 100 == 0:
            print "processing {}".format(cnt)
    return dictCluster, topics

print 'read over'
print x.shape

thres = 0.01
dictCluster, topics = single_pass(x, thres)

i = 0
sum = 0
groupNum = 200


cluster_to_class = {}
cluster_to_class[0] = []
i = 0
for topic in topics:
    if not cluster_to_class.has_key(topic):
        cluster_to_class[topic] = []
    cluster_to_class[topic].append(y[i])
    i = i + 1

purities = []
entropies = []

for i in cluster_to_class.keys():
    sub_yy = cluster_to_class[i]

    max_class = max(set(sub_yy), key=sub_yy.count)
    max_cnt = sub_yy.count(max_class)
    purities.append(max_cnt*1.0/len(topics))

    pij = [sub_yy.count(elem) * 1.0/len(sub_yy) for elem in set(sub_yy)]
    Ei = -np.sum([(p * np.log(p)) for p in pij])
    entropies.append(len(sub_yy) * 1.0/len(topics) * Ei)

purity = np.sum(purities)
entropy = np.sum(entropies)
print(purity)
print(entropy)

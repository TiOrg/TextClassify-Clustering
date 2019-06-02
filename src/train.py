#coding:utf-8 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import os

def getCossim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*(np.linalg.norm(vec2)))


def getMaxSimilarity(vector, centers, variances):
    maxValue = 0
    maxIndex = -1
    for k,center in centers.iteritems():
        join = getCossim(vector, center)/(1 + variances[k])
        oneSimilarity = join
        for kk, c in centers.iteritems():
            if kk != k:
                oneSimilarity = oneSimilarity + getCossim(c, center) * getCossim(vector, c)/(1 + variances[kk])

        if oneSimilarity > maxValue:
            maxValue = oneSimilarity
            maxIndex = k
    return maxIndex, maxValue



def single_pass(vectors, thres):
    topics = []
    dictCluster = {}
    clusterVars = {}
    clusterMeans = {}
    clusterCenter = {}

    numCluster = 0 
    cnt = 0
    topic = 0
    for vector in vectors: 
        if numCluster == 0:
            dictCluster[numCluster] = []
            dictCluster[numCluster].append(vector)

            clusterCenter[numCluster] = vector
            clusterMeans[numCluster] = 0
            clusterVars[numCluster] = 0

            numCluster += 1
    
        else:
            maxIndex, maxValue = getMaxSimilarity(vector, clusterCenter, clusterVars)
            print maxValue
            
            #join the most similar topic
            if maxValue > thres:
                dictCluster[maxIndex].append(vector)

                # print(np.array(dictCluster[maxIndex][0]))
                clusterCenter[maxIndex] = np.mean(np.array(dictCluster[maxIndex]), axis=0)
                # print(clusterCenter[maxIndex])

                newSum = clusterMeans[maxIndex] * len(dictCluster[maxIndex]) + getCossim(vector, clusterCenter[maxIndex])
                # print(newSum)
                clusterMeans[maxIndex] = newSum / (len(dictCluster[maxIndex])+1)
                clusterVars[maxIndex] = np.var([getCossim(clusterCenter[maxIndex], v) for v in dictCluster[maxIndex]])

                # print(0, clusterMeans[0], clusterVars[0])

                topic = maxIndex
            #else create the new topic
            else:
                dictCluster[numCluster] = []
                dictCluster[numCluster].append(vector)

                clusterCenter[numCluster] = vector
                clusterMeans[numCluster] = 0
                clusterVars[numCluster] = 0

                topic = numCluster
                numCluster += 1
        topics.append(topic)
        cnt += 1
        if cnt % 500 == 0:
            print "processing {}".format(cnt)
    return dictCluster, topics


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

print 'read over'

thres = 0.06
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

for c in cluster_to_class.keys():
    print(cluster_to_class[c])

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

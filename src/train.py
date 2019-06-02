#coding:utf-8 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import os
from gensim import corpora, models, similarities, matutils

datapath=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))           

# from sklearn import preprocessing
# from sklearn import metrics
# import matplotlib.pyplot as plt
print 'reading data...'
x=pickle.load(open(datapath + '/data/tfidf_data.pkl','rb'))
y=pickle.load(open(datapath + '/data/train_label.pkl','rb'))

x=np.array(x)
y=np.array(y)

# from sklearn.model_selection import train_test_split  
# x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.15, random_state=33)  

# def try_diff(clf):
# 	clf.fit(x_train,y_train)
# 	score = clf.score(x_test, y_test)
# 	print('score: %f'%score)
# 	predicted = clf.predict(x_test)  
# 	print(metrics.classification_report(y_test,predicted))
# 	# plt.figure()
# 	# plt.plot(np.arange(len(predicted)),y_test,'go-',label='true value')
# 	# plt.plot(np.arange(len(predicted)),predicted,'ro-',label='predict value')
# 	# plt.title('score: %f'%score)
# 	# plt.legend()
# 	# plt.show()


# from sklearn import tree
# DTR = tree.DecisionTreeRegressor()

# from sklearn.svm import LinearSVC  
# SVM= LinearSVC()  

# from sklearn.linear_model import LogisticRegression
# LR=LogisticRegression()

# from sklearn import neighbors
# KNN=neighbors.KNeighborsClassifier()

# from sklearn.naive_bayes import MultinomialNB
# NB=MultinomialNB()

# from sklearn.neural_network import MLPClassifier
# MLP=MLPClassifier()

# from sklearn import ensemble
# RFC = ensemble.RandomForestClassifier(n_estimators=100)  
# ADA = ensemble.AdaBoostClassifier(n_estimators=100)
# GBRT = ensemble.GradientBoostingClassifier(n_estimators=100)



# print('\n\nADA:')
# try_diff(ADA)
# print('\n\nRFC:')
# try_diff(RFC)
# print('\n\nGBRT:')
# try_diff(GBRT)

# print('\n\nDTR:')
# try_diff(DTR)
# print('\n\nMLP:')
# try_diff(MLP)
# print('\n\nSVM:')
# try_diff(SVM)
# print('\n\nLR:')
# try_diff(LR)
# print('\n\nKNN:')
# try_diff(KNN)
# print('\n\nNB:')
# try_diff(NB)

def getCossim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*(np.linalg.norm(vec2)))


def getMaxSimilarity(dictTopic, vector):
    maxValue = 0
    maxIndex = -1
    for k,cluster in dictTopic.iteritems():
        oneSimilarity = np.mean([getCossim(vector, v) for v in cluster])
        if oneSimilarity > maxValue:
            maxValue = oneSimilarity
            maxIndex = k
    return maxIndex, maxValue



def single_pass(vectors, thres):
    dictTopic = {}
    dictCluster = []
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
        dictTopic.append(topic)
        cnt += 1
        if cnt % 100 == 0:
            print "processing {}".format(cnt)
    return dictCluster, dictTopic

print 'read over'
print x.shape

thres = 0.01
dictCluster, dictTopic = single_pass(x, thres)

i = 0
for topic in dictTopic:
    i = i + 1
    print topic,
    if i % 200 == 0:
        print '\n'
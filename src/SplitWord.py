#coding:utf-8 
import jieba  
import os  
import re  
import string  


curpath=os.getcwd()               
readpath=curpath+"/SogouC" 
writepath=curpath+"/SogouCC" 
os.chdir(readpath)  
stopwords = {}.fromkeys([line.rstrip() for line in open(curpath+'/stopwords.txt')])  
class_code = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000024']


def fileWordProcess(contents):  
    wordsList = []
    alist = []  

    for seg in jieba.cut(contents):  
        seg = seg.encode('utf8')
        alist.append(seg)  
        if seg!=' ':                   # remove 空格  
            wordsList.append(seg)      # create 文件词列表  
    file_string = ' '.join(wordsList)              
    return file_string,set(alist)  


for categoryName in class_code:            
    categoryPath_r = os.path.join(readpath,categoryName) # 这个类别的路径  
    categoryPath_w = os.path.join(writepath,categoryName) # 这个类别的路径  

    if not os.path.exists(categoryPath_w):
        os.makedirs(categoryPath_w)
    filesList = os.listdir(categoryPath_r)      # 这个类别内所有文件列表  



    k = 0 #计算每类文件个数
    for filename_r in filesList:  
        contents = open(os.path.join(categoryPath_r,filename_r)).read()  
        wordProcessed,alist = fileWordProcess(contents)       # 内容分词成列表

        temp = open(os.path.join(categoryPath_w,str(k)+".cut"),'wb')  
        temp.write(str(wordProcessed))
        temp.close()
        k += 1
        if(k >= 500):#取前100个文件
            break
#朴素贝叶斯文本分类代码
from numpy import zeros,array
from math import log

def loadDataSet():
    #词条切分后的文档集合，列表每一列代表一个email
    postingList=[['your','mobile','number','is','award','bonus','prize'],
                 ['new','car','and','house','for','my','parents'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['today', 'voda', 'number', 'prize', 'receive', 'award'],
                 ['get', 'new', 'job', 'in', 'company', 'how', 'to', 'get', 'that'],
                 ['free', 'prize', 'buy', 'winner', 'receive', 'cash']]
    #由人工标注的每篇文档的类标签
    classVec=[1,0,0,1,0,1]#1-spam,0-ham
    return postingList,classVec

postingList,classVec=loadDataSet()
print(postingList )
print(classVec)
print('=============================================')
#统计所有文档中出现的词条列表
def createVocabList(dataSet):
    vocabSet=set([])
    #遍历文档集合中的每一篇文档
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)
vocabSet=createVocabList(postingList)
print(vocabSet)
print('================================================')
#根据词条列表中的词条是否在文档中出现（出现1，未出现0），将文档转化为词条向量
def setOfWords2Vec(vocabSet,inputSet):
    #新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec=[0]*len(vocabSet)
    #遍历文档中的每一个词条
    for word in inputSet:
        #如果词条在词条列表中出现
        if word in inputSet:
            #通过列表获取当前的word的索引（下标）
            #将词条向量中的对应下标的项由0改为1
            returnVec[vocabSet.index(word)]=1
        else : print('the word: %s is not in my vocabulary !'%'word')
        #返回inputer转化为的词条向量
        return returnVec
trainMatrix=[setOfWords2Vec(vocabSet,inputSet) for inputSet in postingList]
print(trainMatrix)
print('===============================================')
#训练算法，从词向量计算概率p(w0|ci)...及p（ci）
#trainMatrix:由每篇文档的词条向量组成的文档矩阵
#trainCategory:每篇文档的类标签组成的向量

def trainNB0(trainMatrix,trainCategory):
    #获取文档矩阵中文档的数目
    numTrainDocs=len(trainMatrix)
    #获取词条向量的长度
    numWords=len(trainMatrix[0])
    #所有文档中属于类1所占的比例p(c=1)
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #创建一个长度为词条向量等长的列表
    p0Num=zeros(numWords)#ham
    p1Num=zeros(numWords)#spam
    p0Denom=0.0
    p1Denom=0.0
    #遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        #如果该词条向量对应的标签为1
        if trainCategory[i]==1:
            #统计所有类别为1的词条向量中各个词条出现的次数
            p1Num+=trainMatrix[i]
            #统计类别为1的词条向量中各个词条出现的次数
            #即统计类1所有文档中出现的单词的数目
            p1Denom+=sum(trainMatrix[i])
        else :
            #统计所有类别为0的词条向量中各个词条出现的次数
            p0Num+=trainMatrix[i]
            #统计类别为0的词条向量中出现的所有词条的总数
            #即统计类0所有文档中出现单词的数目
            p0Denom+=sum(trainMatrix[i])
        print(p1Num,p1Denom,p0Num,p0Denom)
        #利用Numpy数组计算p(wi|c1)
        p1Vect=p1Num/p1Denom #为避免下溢出问题，需要改为log（）
        #利用Numpy数组计算p（wi|c0)
        p0Vect=p0Num/p0Denom  #为避免下溢出问题，需要改为log()
        return p0Vect,p1Vect,pAbusive

p0Vect,p1Vect,pAbusive=trainNB0(trainMatrix,classVec)
print(p0Vect)
print(p1Vect)
print(pAbusive)
print('======================================================')
#朴素贝叶斯分类函数
#vec2Classify待测试分类的词条向量
#p0Vec：类别0所有文档中各个词条出现的频数p(wi|c0)
#p1Vec:类别1所有文档中各个词条出现的频数p(wi|c1)
#pClass1:类别为1的文档站文档总数比例
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 'spam'
    else:
        return 'not spam'

testEntry=['love','my','job']
thisDoc=array(setOfWords2Vec(vocabSet,testEntry))
print(testEntry,'classified as:',classifyNB(thisDoc,p0Vect,p1Vect,pAbusive))
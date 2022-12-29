# Codes-used-for-construction-of-the-corporate-sustainability-disclosure-dataset
The codes used for calculation and analysis in construction of the corporate sustainability disclosure dataset
#First. Python code
#Step1. Preprocessing MD&A texts
import pandas as pd
import numpy as np
from tqdm import tqdm
import jieba
import  re
from datetime import datetime
import random
data = pd.read_excel("MD&A texts.xlsx")
data.head(3)
#(1)Removing stop words
def stopwordslist():
    stopwords_1 = [str(line).strip() for line in open('E:\\stopwords-master\\stopwords-master\\baidu_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords_2 = [str(line).strip() for line in open('E:\\stopwords-master\\stopwords-master\\cn_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords_3 = [str(line).strip() for line in open('E:\\stopwords-master\\stopwords-master\\hit_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords_4 = [str(line).strip() for line in open('E:\\stopwords-master\\stopwords-master\\scu_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords_lst = set(stopwords_1 + stopwords_2 + stopwords_3 + stopwords_4)
    stopwords = list(stopwords_lst)
    return stopwords
#(2)Word segmentation
def seg_depart(sentence):
        sentence_depart = jieba.cut(str(sentence).strip())
        stopwords = stopwordslist()
        outstr = ''
        for word in sentence_depart:
            if word not in stopwords:
                if word != '\t' and len(word)>1:
                    outstr += word
                    outstr += " "
        return outstr
data["textnew"] = np.nan
for i in tqdm(range(len(data))):
    line_seg = seg_depart(data.loc[i,"MD&A texts"])
    data.loc[i,'textnew']=line_seg
print("success")
del data["MD&A texts"]
data.head(3)
data.to_excel("MDA texts.xlsx",index = False)
#Step2. Caculate extended words of seed words
import logging
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import jieba.analyse
import codecs
#(1)Importing Data
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if __name__ == '__main__':
    shuju = open('MD&A texts used for analysis.txt', 'rb')
#(2)Using similar function to calculate extended words most related to the seed word a 
model Word2Vec(LineSentence(shuju),sg=1,vector_size=300,window=10,min_count=5,
workers=15,sample=1e-3)
    model.save('2.word2vec')
    print('over ')
    print("Calculate the related word list of a word ")
    word = 'seed word a'
result2 = model.wv.most_similar(word,topn=30)  
#(3)Obtaining extended words of seed word a 
    print("The most relevant words to"+ a +"are:")    
    for item in result2:
        print(item[0], item[1])
print("/n")
#Step3. Caculating the itf.idf of each words in dictionary
import jieba
import pandas as pd
import re
import jieba
from datetime import datetime
import numpy as np
word = "all words in the dictionary of corporate sustainable development "
lst = word.split("、")
print(lst)
#(1)Importing Data
filename = "MD&A texts ready for analysis"  
inputs = pd.read_excel(filename)
# Each word is counted as a column
for i in lst:
    inputs[i] = np.nan
inputs["year"] = np.nan
inputs["corporate"] = np.nan
for i in range(len(inputs)):
    temp_lst = str(inputs.loc[i, "year_corporate"]).split("_")
    inputs.loc[i, "year"] = temp_lst[0]
#(2)Calculating word frequency
for i in range(len(inputs)):
    temp = str(inputs.loc[i, 'textnew']).split(" ")
    inputs.loc[i,"frequency"]=len(temp)
    for j in lst:
        inputs.loc[i, j] = temp.count(j) / len(temp)
def cac_num(lst, data):
    dic = {"word": lst, "num": [0] * len(lst)}
    for i in range(len(lst)):
        for index in range(len(data)):
            if dic["word"][i] in data.loc[index, "textnew"]:
                dic["num"][i] = dic["num"][i] + 1
    return dic
#(3)Calculating “tf.idf” of each word in dictionary
dic_num = cac_num(lst, inputs)
for i in range(len(lst)):
    for ind in range(len(inputs)):
        inputs.loc[ind, lst[i]] =inputs.loc[ind, lst[i]] *np.log(len(inputs) / (1 + dic_num["num"][i]))
inputs.to_excel("tf.idf of each word.xlsx", index=False)

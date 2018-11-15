# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/29 0029 下午 4:17
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用gensim 模块进行词向量的训练
"""
from gensim.models import word2vec
import pickle
with open('allWord.pkl','rb') as f:
    allwords=pickle.load(f)
    allwords=list(allwords.keys())
    print(allwords)
# sentes=[
#     'I am a good student'.split(),
#     'Good good study day day up'.split()
# ]
# print(sentes)
model=word2vec.Word2Vec([allwords],size=300,window=1, min_count=1)
# print(model.wv.word_vec('93-50'))
# print(model.wv.most_similar('93-50',topn=2))
#保存模型得到的词向量到文件
model.wv.save_word2vec_format('word2vec.txt',binary=False)

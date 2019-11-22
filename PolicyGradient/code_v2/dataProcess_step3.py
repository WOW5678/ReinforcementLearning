# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/6 0006 下午 1:57
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import csv
import numpy as np
from collections import Counter

def load_data(filepath):
    '''
    加载数据集
    '''
    patientFeatures=[]
    patientLabels=[]
    with open(filepath) as f:
        reader=csv.reader(f)
        for line in reader:
            feature = line[2][1:-1].replace("'", '').split(' ')
            feature = [item.strip() for item in feature]
            feature = list(set(feature))
            f = ' '.join([item for item in feature]).strip()
            label = line[3][1:-1].replace("'", '').split(';')
            label = [item.strip() for item in label]
            label = list(set(label))
            l = ' '.join([item for item in label]).strip()
            patientFeatures.append(f)
            patientLabels.append(l)
        return patientFeatures, patientLabels
def build_vocab(sentences,max_vocab_size=None):
    word_counter=Counter()
    vocab=dict()
    reverse_vocab=dict()

    for sent in sentences:
        tokens=sent.split()
        word_counter.update(tokens)
    if max_vocab_size is None:
        max_vocab_size=len(word_counter)

    vocab['PAD']=0
    vocab['UNK']=1
    vocab_idx=2
    for key,value in word_counter.most_common(max_vocab_size):
        vocab[key]=vocab_idx
        vocab_idx+=1
    for key,value in vocab.items():
        reverse_vocab[value]=key
    return vocab,reverse_vocab,max_vocab_size

def build_vocab_label(sentences):
    token2id={}
    id2token={}
    for sent in sentences:
        tokens = sent.split()
        for token in tokens:
            if token not in token2id:
                token2id[token]=len(token2id)

    for key, value in token2id.items():
        id2token[value] = key
    return token2id,id2token

def transform2ids(patientFeatures,feature2id,max_len=70):
    featureIds=[]
    for row in patientFeatures:
        tokens=row.split()
        ids=[feature2id.get('PAD') for i in range(max_len)]
        for i in range(min(max_len,len(tokens))):
            ids[i]=feature2id.get(tokens[i],feature2id.get('UNK'))
        featureIds.append(ids)
    return featureIds

def labelEncoder(patientLabels,label2id):
    labelIds=[]
    for row in patientLabels:
        tokens=row.split()
        temp=np.zeros(len(label2id))
        for token in tokens:
            temp[label2id.get(token)]=1
        labelIds.append(temp)
    return labelIds

def labelEncoder_2(patientLabels,label2id):
    labelIds=[]
    for row in patientLabels:
        tokens=row.split()
        temp=[]
        for token in tokens:
            temp.append(label2id.get(token))
        labelIds.append(temp)
    return labelIds

'''
if __name__ == '__main__':
    # patientFeatures:['头晕 疼痛 右下肢'，'....’]
    # patientLabels:['鼻骨骨折 面部外伤 高血压','....']
    patientFeatures, patientLabels=load_data(filepath='../data/patientFeatures_test.csv')
    feature2id, id2feature, feature_vocab_size=build_vocab(patientFeatures,max_vocab_size=200)
    label2id, id2label, label_vocab_size=build_vocab(patientLabels)
    features=transform2ids()
    print('features:',features)
    labels=labelEncoder()
    print('labels:',labels)
'''
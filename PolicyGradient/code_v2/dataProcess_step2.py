# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/22 0022 下午 3:33
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 处理MIMIC-III数据集
"""
import csv
import numpy as np
import pandas as pd
import datasets
import tqdm
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from scipy.sparse import csr_matrix
from dataProcess_step1 import *
from collections import Counter
import operator
import re

def concat_text_label(label_file,note_file,newfile):
    f_new=open(newfile,'w',newline='')
    writer=csv.writer(f_new)
    print(writer)
    writer.writerow(['SUBJECT_ID','HADM_ID','TEXT','LABEL'])
    with open(label_file,'r') as f_label:
        reader_label=csv.reader(f_label)
        with open(note_file,'r') as f_note:
            reader_note=csv.reader(f_note)
            #进行拼接的依据是subj_id和hadm_id这两个字段
            for row_label in reader_label:
                if row_label[0]!='ROW_ID':
                    #print('row_label:',row_label)
                    temp=[row_label[1],row_label[2]]
                    for row_note in reader_note:
                       print('row_note:',row_note)
                       if row_note[0]==row_label[1] and row_note[1]==row_label[2]:
                           temp.append(row_note[3])
                           temp.append(row_label[4])
                    print('temp:',temp)
                    if len(temp)==4:
                        #print(writer)
                        writer.writerow(temp)
    #writer.close()
    f_new.close()

def filter_tokens(filename):
    f_w=open('../filtered_data.csv','w',newline='')
    writer=csv.writer(f_w)

    with open(filename,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            if row[0]!='SUBJECT_ID':
                raw_dsum = re.sub(r'\[[^\]]+\]', '', row[2])
                raw_dsum = re.sub(r'admission date', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'discharge date', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'date of birth', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'sex', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'service', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'dictated by.*$', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'completed by.*$', '', raw_dsum, flags=re.I)
                raw_dsum = re.sub(r'signed electronically by.*$', '', raw_dsum, flags=re.I)
                print(raw_dsum)
                writer.writerow([row[0],row[1],raw_dsum,row[3]])
    f_w.close()



'''
# data processing
# combine diagnosis and procedure codes and reformat them
dfdiag=pd.read_csv('../MIMIC-III/DIAGNOSES_ICD.csv')
dfdiag['absolute_code']=dfdiag.apply(lambda row:str(datasets.reformat(str(row[4]),True)),axis=1)

dfdiag.to_csv('../MIMIC-III/ALL_CODES.csv', index=False,
               columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
               header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])

# How many codes are there?
df=pd.read_csv('../MIMIC-III/ALL_CODES.csv',dtype={'ICD9_CODE':str})
print(len(df['ICD9_CODE'].unique())) #8994

# Tokenize and preprocess raw text
# Select only discharge summaries and their addenda
# remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
# lowercase all tokens

# 对notes文件进行初步处理之后写出的文件
disch_full_file=write_discharge_summaries(out_file='../MIMIC-III/disch_full.csv')

# Read the notes and see what kind of data we're working with
df=pd.read_csv('../MIMIC-III/disch_full.csv')
print(len(df['HADM_ID'].unique()))#52726

# tokens and types
#对discharge summary进行tokenizer
types=set()
num_tok=0
for row in df.itertuples():
    # row 是一个pands对象，Pandas(Index=645, SUBJECT_ID=28521, HADM_ID=126104, CHARTTIME=nan, TEXT=。。。。）
    for w in row[4].split():
        types.add(w)
        num_tok+=1
print('Num types,',len(types)) #150854
print('Num tokens',str(num_tok)) #79801387

# sorted by SUBJECT_ID和HADM_ID to make a correspondence with the MIMIC label file
df=df.sort_values(['SUBJECT_ID','HADM_ID'])
#Sort the label file by the same
df1=pd.read_csv('../MIMIC-III/ALL_CODES.csv')
df1=df1.sort_values(['SUBJECT_ID','HADM_ID'])
#(52726, 58976)
print(len(df['HADM_ID'].unique()),len(df1['HADM_ID'].unique()))


#将文本和标签对应起来 并写入一个文件中
concat_text_label('../MIMIC-III/ALL_CODES.csv','../MIMIC-III/disch_full.csv','../MIMIC-III/note_label.csv')
'''

# 过滤掉无用的单词
filter_tokens('../notes_labeled.csv')

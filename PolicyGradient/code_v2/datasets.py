# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/5 0005 下午 6:46
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 定义一些数据加载方法

"""
from collections import Counter
import csv
import math
import numpy as np
import sys

from constants import *

class Bacth(object):
    '''

    This class and the data_generator coulud probably be replaced with a pytorch dataloader
    '''
    def __init__(self,desc_mebed):
        self.docs=[]
        self.labels=[]
        self.hadm_ids=[]
        self.code_set=set()
        self.length=0
        self.max_length=MAX_LENGTH
        self.desc_embed=desc_mebed
        self.descs=[]

    def add_instance(self,row,ind2c,c2ind,w2ind,dv_dict,num_labels):
        '''
        Makes an instance to add to this batch form given row data,with a bunch of lookups
        :param row:
        :param ind2c:
        :param c2ind:
        :param w2ind:
        :param dv_dict:
        :param num_labels:
        :return:
        '''
        labwels = set()
        hadm_id=int(row[1])
        text=row[2]
        length=int(row[4])
        cur_code_set=set()
        labels_idx=np.zeros(num_labels)
        labelled=False
        desc_vecs=[]

        for l in row[3].split(';'):
            for i in c2ind.keys():
                code=int(c2ind[i])
                labels_idx[code]=1
                cur_code_set.add(code)
                labelled=True
            if not labelled:
                return
            if self.desc_embed:
                for code in cur_code_set:
                    l=ind2c[code]
                    if  l in dv_dict.keys():
                        desc_vecs.append(dv_dict[l][:])
                    else:
                        desc_vecs.append([len(w2ind)+1])

def reformat(code,is_diag):
    '''
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    :param code:
    :param id_diag:
    :return:
    '''
    code=''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code)>4:
                code=code[:4]+'.'+code[4:]
        else:
            if len(code)>3:
                code=code[:3]+'.'+code[3:]
    else:
        code=code[:2]+'.'+code[2:]
    #print('code:', code)
    return code





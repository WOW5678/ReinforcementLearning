import numpy as np
import tensorflow as tf
import json, random
import pickle

class DataManager(object):
    def __init__(self, dataset):
        '''
        Read the data from dir "dataset"
        '''
        self.origin = {}
        for fname in ['train', 'dev', 'test']:
            data = []
            for line in open('%s/%s.res' % (dataset, fname)):
                s = json.loads(line.strip())
                if len(s) > 0:
                    data.append(s)
            self.origin[fname] = data

    def getword(self):
        '''
        Get the words that appear in the data.
        Sorted by the times it appears.
        {'ok': 1, 'how': 2, ...}
        Never run this function twice.
        '''
        wordcount = {}
        '''
        递归的调用
        '''
        def dfs(node):
            if node.get('children'):
                dfs(node['children'][0])
                dfs(node['children'][1])
            else:
                word = node['word'].lower()
                wordcount[word] = wordcount.get(word, 0) + 1

        for fname in ['train', 'dev', 'test']:
            for sent in self.origin[fname]:
                dfs(sent)
        # with open('../WordVector/allWord.pkl','wb') as f:
        #     pickle.dump(wordcount,f)

        words = wordcount.items()
        #words.sorted(key = lambda x : x[1], reverse = True)
        words=sorted(words,key=lambda x:x[1],reverse=True)
        #print('words:',words)
        self.words = words
        self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}
        return self.wordlist
    
    def getdata(self, grained, maxlenth):
        '''
        Get all the data, divided into (train,dev,test).
        For every sentence, {'words':[1,3,5,...], 'solution': [0,1,0,0,0]}
        For each data, [sentence1, sentence2, ...]
        Never run this function twice.
        '''
        def one_hot_vector(r):
            s = np.zeros(grained, dtype=np.float32)
            # print('s:',s)
            s[r] += 1.0
            return s
        '''
        递归地调用宽度优先函数
        '''
        def dfs(node, words):
            if node.get('children'):
                dfs(node['children'][0], words)
                dfs(node['children'][1], words)
            else:
                word = self.wordlist[node['word'].lower()]
                words.append(word)
        self.getword()
        self.data = {}
        for fname in ['train', 'dev', 'test']:
            self.data[fname] = []
            for sent in self.origin[fname]:
                words = []
                dfs(sent, words)
                lens = len(words)
                if maxlenth <= lens:
                    #print (lens)
                    words=words[:maxlenth]
                    lens = maxlenth
                else:
                    words += [0] * (maxlenth - lens)

                # print('int(sent[rating]):',int(sent['rating']))
                solution = one_hot_vector(int(sent['rating']))
                now = {'words': np.array(words), \
                        'solution': solution,\
                        'lenth': lens}
                self.data[fname].append(now)
        return self.data['train'], self.data['dev'], self.data['test']
    
    def get_wordvector(self, name):
        fr = open(name)
        #n, dim = map(int, fr.readline().split())
        n,dim=fr.readline().split()
        #self.wv中key为单词id,值为对应的向量
        self.wv = {}
        for i in range(int(n)):
            vec = fr.readline().split()
            word = vec[0].lower()
            #vec = map(float, vec[1:])
            #print('vec[1:]:',vec[1:])
            try:
                vec=[float(item) for item in vec[1:]]
                if self.wordlist.get(word):
                    self.wv[self.wordlist[word]] = vec
            except:
                pass
        self.wordvector = []
        losscnt = 0 # 统计不在wordlist中的词的个数
        for i in range(len(self.wordlist)):
            if self.wv.get(i) and len(self.wv.get(i))==int(dim):
                #print('len(wv[i]):',len(self.wv.get(i)))
                self.wordvector.append(self.wv[i])
            else:
                losscnt += 1
                self.wordvector.append(np.random.uniform(-0.1,0.1,[int(dim)]))

        self.wordvector = np.array(self.wordvector, dtype=np.float32)
        print (losscnt, "words not find in wordvector")
        #print (len(self.wordvector), "words in total")

        return self.wordvector

# datamanager = DataManager("../AGnews")
# train_data, test_data, dev_data = datamanager.getdata(4, 70)
# # for i in test_data[:10]:
# #     print(len(i['words']))
# word_vector = datamanager.get_wordvector('../WordVector/word2vec.txt')
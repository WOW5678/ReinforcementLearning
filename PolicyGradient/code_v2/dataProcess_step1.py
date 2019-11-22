# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/5 0005 下午 9:29
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import gensim.models.word2vec as w2v
import csv
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import gensim.models
from constants import *
import datasets
import operator
import tqdm


def write_discharge_summaries(out_file):
    '''

    :param out_file:
    :return:
    '''
    # retain only alphanumeric
    tokenizer = RegexpTokenizer(r'\w+')

    note_file='../MIMIC-III/NOTEEVENTS.csv'
    print('processing notes file')
    with open(note_file,'r') as csvfile:
        with open(out_file,'w') as outfile:
            print('writing to %s'%out_file)
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT'])+'\n')
            notereader=csv.reader(csvfile)

            # Header
            next(notereader)
            i=0
            for line in notereader:
                subj=int(line[1])
                category=line[6]
                if category=='Discharge summary':
                    note=line[10]
                    # tokenize,lowercase and remove numberics
                    tokens=[t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text='"'+' '.join(tokens)+'"'
                    outfile.write(','.join([line[1],line[2], line[4], text])+'\n')
                i+=1
    return out_file

def concat_data(labelsfile,notes_file):
    '''

    :param labelsfile:
    :param notes_file:
    :return:
    '''
    with open(labelsfile,'r') as lf:
        print('Concatnating')
        with open(notes_file,'r') as notes_file:
            outfilename='../MIMIC-III/notes_labeled.csv'
            with open(outfilename,'w',newline='') as outfile:
                w=csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen=next_labels(lf)
                notes_gen=next_notes(notes_file)

                for i,(subj_id,text,hadm_id) in enumerate(notes_gen):
                    if i%10000==0:
                        print(str(i)+'done')
                    cur_subj,cur_labels,cur_hadm=next(labels_gen)

                    if cur_hadm ==hadm_id:
                        w.writerow([subj_id,str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print('Could not find matching hadm_id, data is probably not sorted correctly')
                        break
    return outfilename

def next_labels(labelsfile):
    '''
    Generate for label sets from the label file
    '''
    labels_reader=csv.reader(labelsfile)
    #header
    next(labels_reader)

    first_label_line=next(labels_reader)

    cur_subj=int(first_label_line[0])
    cur_hadm=int(first_label_line[1])
    cur_labels=[first_label_line[2]]

    for row in labels_reader:
        subj_id=int(row[0])
        hadm_id=int(row[1])
        code=row[2]
        # keep reading until you hit a new hadm id
        if hadm_id!=cur_hadm or subj_id!=cur_subj:
            yield  cur_subj,cur_labels,cur_hadm
            cur_labels=[code]
            cur_subj=subj_id
            cur_hadm=hadm_id
        else:
            cur_labels.append(code)
    yield cur_subj,cur_labels,cur_hadm

def next_notes(notesfile):
    '''
    Generate for notes from the notes file
    This will also concatenate discharge summaries
    :param notesfile:
    :return:
    '''
    nr=csv.reader(notesfile)
    #header
    next(nr)
    first_note=next(nr)

    cur_subj=int(first_note[0])
    cur_hadm=int(first_note[1])
    cur_text=first_note[3]

    for row in nr:
        subj_id=int(row[0])
        hadm_id=int(row[1])
        text=row[3]
        #keep reading until you hit a new hadm id
        if hadm_id!=cur_hadm or subj_id!=cur_subj:
            yield cur_subj,cur_text,cur_hadm
            cur_text=text
            cur_subj=subj_id
            cur_hadm=hadm_id
        else:
            cur_text+=' '+text
    yield  cur_subj,cur_text,cur_hadm

def split_data(labeledfile):
    print('Spliting data')
    # create and writer headers for train,dev,test
    train_name='../MIMIC-III/train_split.csv'
    dev_name='../MIMIC-III/dev_split.csv'
    test_name='../MIMIC-III/test_split.csv'
    train_file=open(train_name,'w')
    dev_file=open(dev_name,'w')
    test_file=open(test_name,'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])+'\n')
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])+'\n')
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])+'\n')

    hadm_ids={}

    # read in train,dev,test splits
    for splt in ['train','dev','test']:
        hadm_ids[splt]=set()
        with open('../MIMIC-III/%s_full_hadm_ids.csv'%splt,'r')as  f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile,'r') as lf:
        reader=csv.reader(lf)
        next(reader)
        i=0

        cur_hadm=0
        for row in reader:
            # filter next,write to file according to train/dev/test split
            if i%10000==0:
                print(str(i)+'read')
            #print('row[1]:',row[1])
            hadm_id=row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row)+'\n')
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row)+'\n')
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row)+'\n')
            i+=1
        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name


def build_vocab(vocab_min,infile,vocab_filename):
    """

    :param vocab_min:how many documents a word must appear in to be kept
    :param infile: (training) data file to build vocabulary from
    :param vocab_filename: name for the file to output
    :return:
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        #header
        next(reader)

        #0. read in data
        print("reading in data...")
        #holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        # also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))

        # 2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        # drop those rows
        C = C[inds, :]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")

class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())


def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.w2v" % (Y)
    sentences = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, workers=4, iter=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file


def gensim_to_embeddings(wv_file, vocab_file, outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)

def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index2word[0])) ))
    words = [PAD_CHAR]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        #UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


def vocab_index_descriptions(vocab_file, vectors_file):
    # load lookups
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}
    desc_dict = datasets.load_code_descriptions()

    tokenizer = RegexpTokenizer(r'\w+')

    with open(vectors_file, 'w') as of:
        w = csv.writer(of, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in tqdm(desc_dict.items()):
            # same preprocessing steps as in get_discharge_summaries
            tokens = [t.lower() for t in tokenizer.tokenize(desc) if not t.isnumeric()]
            inds = [w2ind[t] if t in w2ind.keys() else len(w2ind) + 1 for t in tokens]
            w.writerow([code] + [str(i) for i in inds])
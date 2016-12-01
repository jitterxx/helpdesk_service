#!/usr/bin/python3 -t
# coding: utf8


import uuid



import tensorflow as tf
import numpy as np
import gensim
import re
import os
import nltk

import sys
sys.path.extend(['..'])

from helpers import data_helpers_test

"""
def clean_str(string):

    #print 'Raw sting: ', string, type(string), len(string)


    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub("[\s*|\d*]", " ", string, re.UNICODE)

    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)


    #string = re.sub(r"[\\n]+", r" ", string, re.U)
    #string = re.sub(r"[\\r]+", r" ", string, re.U)

    #string = re.sub(r'[\s]+', r' ', string, re.U)
    #string = re.sub(r'[\w]+', r' ', string, re.U)
    #string = re.sub(r"[\d]+", r" ", string, re.U)
    #string = re.sub(r"[,\.:;'\\\"`!@/\?\[\]\(\)\{\}_\*&%\$#=\+~<>]+", r" ", string, re.U)
    #string = re.sub(r'[\-{2,}]', r' ', string, re.U)
    #string = re.sub(r'[\s\-\s]', r' ', string, re.U)
    #string = re.sub(r'[_{2,}]', r' ', string, re.U)

    #string = re.sub("[\d]", " ", string, re.UNICODE|re.M)
    #string = re.sub("[,.:;'\"`!@?/\\|\[\](){}_/*&/%$#=+~]", " ", string, re.UNICODE|re.M)

    #string2 = nltk.tokenize.word_tokenize(string)
    #print("\n NLTK tokens")
    #print string2

    string = string.decode('utf-8').lower()

    string = re.sub(u'[^абвгджзёеыйиклмнуопрстфчхшщцэюяьъqwertyuiopasdfghjklzxcvbnm]+', ' ', string.decode('utf-8'), re.U)
    string = re.sub(u"\s{2,}", " ", string.decode('utf-8'), re.U)

    #print('\nClear sting: {}'.format(string))
    #print type(string), len(string)
    #raw_input()

    return string.split(' ')
"""

def load_data_and_labels(data_dir=None):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    data = list()

    # Load data from files
    print("{}/".format(data_dir))
    file_list = list()
    for root, dirs, f_list in os.walk("{}/".format(data_dir)):
        for oo in f_list:
            if 'class.nfo' != oo:
                file_list.append('{}/{}'.format(data_dir, oo))

    for ff in file_list:
        f = open(ff, 'r')
        ss = f.readlines()
        data = data + ss

    # Split by words
    data = [data_helpers_test.clean_str_new(sent) for sent in data]

    return data

"""
sentences = load_data_and_labels(data_dir='./data/train')


print '\n Рассчитываем word2vec...'
model = gensim.models.Word2Vec(sentences, min_count=1, size=300, workers=1, iter=10, window=7)

model.save('./data/word2vec_model')

"""


from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import shutil

Base = automap_base()

# engine, suppose it has two tables 'user' and 'address' set up
sql_uri = "mysql://%s:%s@%s:%s/%s?charset=utf8" % ('conparser', 'Qazcde123', 'localhost', '3306', 'otrs')
engine = create_engine(sql_uri)

# reflect the tables
Base.prepare(engine, reflect=True)

# mapped classes are now created with names by default
# matching that of the table name.
Ticket = Base.classes.ticket
Article = Base.classes.article
Field = Base.classes.dynamic_field_value
Value = Base.classes.ut_dynamic_field_value

session = Session(engine)

data = list()

resp = session.query(Article.a_body).all()

try:
    print("Кол-во примеров: ", len(resp))
except Exception:
    pass

for one, in resp:
    # text = re.split('[\r|\n]+', one)
    word_list = [s for s in data_helpers_test.clean_str_new(one).split(' ')]
    # print('text: ', word_list)
    data.append(word_list)



# Split by words
#data = [data_helpers_test.clean_str_new(sent) for sent in data]
#print('data: ', data)

print('\n Рассчитываем word2vec - size=300, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=300, workers=2, window=7, iter=10)
model.save('./data/word2vec_model_dim300_new_data_helper_w7_i10')

print('\n Рассчитываем word2vec - size=400, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=400, workers=2, window=7, iter=10)
model.save('./data/word2vec_model_dim400_new_data_helper_w7_i10')

print('\n Рассчитываем word2vec - size=500, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=500, workers=2, window=7, iter=10)
model.save('./data/word2vec_model_dim500_new_data_helper_w7_i10')


session.close()


word = 'заявка'
print('\n Ближайщие к слову {}: '.format(word))
try:
    for one in model.similar_by_word(word):
        print('{} - {}'.format(one[0], one[1]))
except Exception as e:
    print("Слова - {}, нет в словаре.".format(word))

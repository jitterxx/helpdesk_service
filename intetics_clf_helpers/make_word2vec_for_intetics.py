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

from helpers import data_helpers_intetics


from helpers import data_helpers_conparser
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import shutil

Base = automap_base()

# engine, suppose it has two tables 'user' and 'address' set up
sql_uri = "mysql://%s:%s@%s:%s/%s?charset=utf8" % ('root', 'OO00zZOK', '127.0.0.1', '33067', 'conparser')
engine = create_engine(sql_uri)

# reflect the tables
Base.prepare(engine, reflect=True)

# mapped classes are now created with names by default
# matching that of the table name.
Msg = Base.classes.email_cleared_data

session = Session(engine)

data = list()

resp = session.query(Msg).all()

try:
    print("Кол-во примеров: ", len(resp))
except Exception:
    pass

for one in resp:
    text = '{}\n\n{}'.format(one.message_title, one.message_text)
    word_list = [s for s in data_helpers_intetics.clean_str_orig(text).split(' ')]
    # print('word_list: ', word_list)
    data.append(word_list)
    # input()


PATH = "{}/{}_train_data".format(sys.argv[1], sys.argv[2])
fname = {'conflict': 'rt-polarity.neg', 'normal': 'rt-polarity.pos'}

for fn in list(fname.values()):
    for text in open("{}/{}".format(PATH, fn), 'r', encoding="latin-1").readlines():
        word_list = [s for s in data_helpers_intetics.clean_str_orig(text).split(' ')]
        # print('word_list: ', word_list)
        data.append(word_list)
        # input()


# Split by words
#data = [data_helpers_test.clean_str_new(sent) for sent in data]
#print('data: ', data)

print('\n Рассчитываем word2vec - size=300, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=300, workers=2, window=7, iter=10)
model.save('./data/intetics_word2vec_model_dim300_new_data_helper_w7_i10')

print('\n Рассчитываем word2vec - size=400, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=400, workers=2, window=7, iter=10)
model.save('./data/intetics_word2vec_model_dim400_new_data_helper_w7_i10')

print('\n Рассчитываем word2vec - size=500, workers=2, window=7, iter=10')
model = gensim.models.Word2Vec(data, min_count=1, size=500, workers=2, window=7, iter=10)
model.save('./data/intetics_word2vec_model_dim500_new_data_helper_w7_i10')

word = 'client'
print('\n Ближайщие к слову {}: '.format(word))
try:
    for one in model.similar_by_word(word):
        print('{} - {}'.format(one[0], one[1]))
except Exception as e:
    print("Слова - {}, нет в словаре.".format(word))

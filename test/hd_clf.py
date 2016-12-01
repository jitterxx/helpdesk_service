#!/usr/bin/python -t
# coding: utf8

"""
Классификация текстов(документов), сообщений и т.д.

1. Содержит модели классификаторов
2. Для каждой модели определены методы загрузки данных из базы, тренировки и извлечения признаков

"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.extend(['..'])

import re
import os
import json
from sqlalchemy import func

import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import pymorphy2

morph = pymorphy2.MorphAnalyzer()


STOP_WORDS = ["как", "или", "который", "которых", "тот", "около", "они", "для", "Для", "Это", "это", "При", "при",
              "Кроме", "того", "чем", "под", "них", "его", "лат", "Также", "также", "этой", "этого",
              "com", "вам", "Вам", "Вами", "вами", "Вас", "вас", "ваше", "Ваше", "Все", "все",
              "добрый", "день", "спасибо", "здравствуйте", "добрый день", "утро", "коллеги"]

CATS = ['00', '01', '11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '31',
        '33', '41', '42', '43', '44', '45', '46', '47', '51']

data_folder = 'hd_train_data'

def specfeatures_t2(entry):

    tt = ""

    tt += entry.message_title + "\n"
    tt += entry.message_text + "\n"

    return tt


def tokenizer_t2(entry):

    # print entry

    splitter = re.compile('\\W*', re.UNICODE)
    find_n = re.compile('\d+', re.UNICODE)
    f = dict()

    result = splitter.split(entry)

    for i in range(0, len(result) - 1):
        one = result[i].lower()

        #number = find_n.search(one)

        #if number:
        #    break

        # print one
        if f.get(one):
            f[one] += 1
        else:
            f[one] = 1

        pair = '{} {}'.format(result[i].lower(), result[i + 1].lower())
        # print pair
        if f.get(pair):
            f[pair] += 1
        else:
            f[pair] = 1

    # for a, b in f.iteritems():
    #   print a, " - ", b

    # raw_input()

    return f


class ClassifierNew(object):
    clf = None
    outlier = None
    vectorizer = None
    scaler = None
    debug = False
    test_data = None
    test_answer = None

    def init_and_fit_files(self, debug=False):
        """
        Инициализация классификаторов и тренировка.
        Чтение данных из файлов.

        :return:
        """

        session = Session()
        self.debug = debug

        cats = CATS

        train = list()
        answer = list()
        count = 0

        class msg_data(object):
            message_text = ''
            message_title = ''
            category = ''

        for current_cat in cats:
            print("{}/{}/".format(data_folder, current_cat))
            file_list = list()
            for root, dirs, f_list in os.walk("{}/{}/".format(data_folder, current_cat)):
                for oo in f_list:
                    if 'class.nfo' != oo:
                        file_list.append('{}/{}/{}'.format(data_folder, current_cat, oo))

            print("Готовим категорию: {} - {} сообщений".format(current_cat, len(file_list)))

            for ff in file_list:
                f = open(ff, 'r')
                ss = f.read()
                new = msg_data()
                new.message_text = ss

                train.append(new)
                answer.append(current_cat)

        from sklearn.model_selection import train_test_split

        train, self.test_data, answer, self.test_answer = train_test_split(train, answer, test_size=0.15, random_state=42)

        print("Count Train: {}".format(len(train)))
        print('Count Test: {}'.format(len(self.test_data)))

        # Готовим векторизатор
        use_hashing = False
        t0 = time()


        if use_hashing:
            #vectorizer = HashingVectorizer(stop_words=STOP_WORDS, analyzer='word', non_negative=True, n_features=60000,
            #                               tokenizer=mytoken, preprocessor=specfeatures_new)
            #vectorizer = HashingVectorizer(stop_words=CPO.STOP_WORDS, analyzer='word', non_negative=True, n_features=10000,
            #                               tokenizer=CPO.mytoken, preprocessor=CPO.specfeatures_new2,
            #                               norm='l1')
            #vectorizer = HashingVectorizer(analyzer='word', n_features=10000, non_negative=True, norm='l1',
            #                               tokenizer=CPO.mytoken, preprocessor=CPO.specfeatures_new2)

            vectorizer = HashingVectorizer(analyzer='char', n_features=10000, non_negative=True, norm='l1',
                                           preprocessor=specfeatures_t2)


            X_train = vectorizer.transform(train)
        else:
            #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1, stop_words=CPO.STOP_WORDS, analyzer='word',
            #                             tokenizer=CPO.mytoken, preprocessor=CPO.features_extractor2)

            #vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=CPO.STOP_WORDS, analyzer='word',
            #                             preprocessor=specfeatures_t2, lowercase=True, norm='l1',
            #                             max_df=0.5, min_df=0.02)

            vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=STOP_WORDS, analyzer='word',
                                         preprocessor=specfeatures_t2, lowercase=True, norm='l1',
                                         max_df=0.7, min_df=0.01,
                                         tokenizer=tokenizer_t2)

            #vectorizer = TfidfVectorizer(analyzer='char',
            #                             preprocessor=specfeatures_t2, ngram_range=(3, 5), max_features=10000)

            X_train = vectorizer.fit_transform(train)

            for one in vectorizer.get_feature_names():
                # print one
                pass

        scaler = StandardScaler(with_mean=False).fit(X_train)
        #X_train = scaler.transform(X_train)

        self.scaler = scaler
        self.vectorizer = vectorizer

        if self.debug:
            duration = time() - t0
            print "TRAIN data"
            print("done in %fs at %0.3fText/s" % (duration, len(train) / duration))
            print("n_samples: %d, n_features: %d" % X_train.shape)
            print "\n"

        # Создаем классификаторы
        self.clf = list()

        """
        self.clf.append(MLPClassifier(solver='sgd', verbose=False, max_iter=500))


        self.clf.append(
            GridSearchCV(MLPClassifier(verbose=False),
                         param_grid={'alpha': [0.01, 0.001, 0.0001],
                                     'hidden_layer_sizes': [
                                         (1000, 500, 100),
                                         (1000, 500), (1000, 1000),
                                         (500, 500), (500, 300), (500, 100) ],
                                     'activation': ['tanh', 'relu'],
                                     'max_iter': [5000, 10000, 15000],
                                     'solver': ['lbfgs']})
        )
        """

        # use_hashing = False
        # {'alpha': 0.0001, 'activation': 'tanh', 'hidden_layer_sizes': (1000, 500)} = score: 0.617486338798
        # {'alpha': 0.001, 'activation': 'relu', 'hidden_layer_sizes': (500, 500)} = 0.612021857923
        # {'alpha': 0.001, 'activation': 'tanh', 'hidden_layer_sizes': (1000, 500), 'max_iter': 5000, 'solver': 'lbfgs'} = 0.73
        # {'alpha': 0.0001, 'activation': 'tanh', 'max_iter': 10000, 'solver': 'lbfgs', 'hidden_layer_sizes': (1000, 500, 100)} = 0.66

        # use_hashing = True
        # {'alpha': 0.001, 'activation': 'tanh', 'max_iter': 500, 'solver': 'lbfgs', 'hidden_layer_sizes': (100,)} = 0.672131147541
        # {'alpha': 0.0005, 'activation': 'relu', 'max_iter': 2000, 'solver': 'lbfgs', 'hidden_layer_sizes': (100,)} = 0.672131147541



        """
        self.clf.append(
            GridSearchCV(MLPClassifier(solver='lbfgs', verbose=False),
                         param_grid={'alpha': [0.05, 0.01],
                                     'hidden_layer_sizes': [(100,), (300,)],
                                     'activation': ['relu']})
        )
        """

        # self.clf.append(MLPClassifier(alpha=0.001, activation='tanh', max_iter=500, solver='lbfgs', hidden_layer_sizes=(100,)))
        self.clf.append(MLPClassifier(alpha=0.001, activation='tanh', max_iter=5000, solver='lbfgs', hidden_layer_sizes=(1000, 500)))
        # self.clf.append(MLPClassifier(alpha=0.001, activation='tanh', max_iter=5000, solver='lbfgs', hidden_layer_sizes=(1000, 500)))
        self.clf.append(MultinomialNB(alpha=0.1))
        self.clf.append(BernoulliNB(binarize=0.0, alpha=0.1))

        # self.clf.append(BernoulliNB(alpha=0.1, binarize=0.0))

        """
        self.clf.append(
            GridSearchCV(
                MultinomialNB(),
                param_grid={'alpha': [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
            )
        )
        # hashing = True, scaling = True
        # {'alpha': 0.1} = 0.644808743169

        self.clf.append(
            GridSearchCV(
                BernoulliNB(binarize=0.0),
                param_grid={'alpha': [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
            )
        )
        """
        # hashing = True, scaling = True
        # {'alpha': 1e-05} = 0.606557377049

        t0 = time()
        # Тренируем классификатор
        for one in self.clf:
            print("Training: {}".format(one))
            one.fit(X_train, answer)
            if isinstance(one, GridSearchCV):
                print("Оценка: ")
                print one.cv_results_['rank_test_score']
                print one.best_params_
                print one.best_score_
        train_time = time() - t0
        if self.debug:
            print("train time: %0.3fs" % train_time)


    def classify_new2(self, data=None, debug=False):
        """
        Классификация образца классификатором.
        Проверка результата классификации детектором аномалий.
        Если детектор и классификатор определяют образец как аномалию (т.е. - conflict), соглашаемся.
        Если детектор считаем аномалией, а классфикатор нет, возращаем результат детектора.

        :return:
        """

        test = [data]
        X_test = self.vectorizer.transform(test)
        #X_test = self.scaler.transform(self.vectorizer.transform(test))

        pred = list()
        complex_pred = dict()
        for one in self.clf:
            p = one.predict(X_test)
            pred.append(p)
            if p[0] in complex_pred.keys():
                complex_pred[p[0]] += 1
            else:
                complex_pred[p[0]] = 1

        if self.debug:
            print("Классификация: {}".format(pred))
            print("Комплексный итог: {}".format(complex_pred))

        for one in complex_pred.keys():
            if complex_pred[one] >= 2:
                return one, "{0}-1:{0}-1:{0}-1".format(one)

        """
        if (float(complex_pred) / 3) > 0.5:
            return "normal", "normal-1:" + "-0.5:".join(pred) + "-0.5"
        else:
            return "conflict", "conflict-1:" + "-0.5:".join(pred) + "-0.5"
        """

    def score(self, test_x=None, test_y=None):

        x_test1 = self.vectorizer.transform(test_x)

        for one in self.clf:
            print one.score(x_test1, test_y)
            if isinstance(one, MLPClassifier):
                print [coef.shape for coef in one.coefs_]

    def dump(self, dir='.'):
        joblib.dump(self.vectorizer, '{}/vectorizer.pkl'.format(dir))
        joblib.dump(self.scaler, '{}/scaler.pkl'.format(dir))
        for i in range(len(self.clf)):
            joblib.dump(self.clf[i], '{}/{}_clf.pkl'.format(dir, i))



if __name__ == '__main__':

    session = Session()
    import requests



    predictor = ClassifierNew()
    predictor.init_and_fit_files(debug=True)

    # Загружаем тестовые данные
    print("Testing...\n")
    predictor.score(predictor.test_data, predictor.test_answer)

    for i in range(len(predictor.test_data)):
        print "*"*30
        print predictor.test_data[i].message_text
        print("Answer: {}".format(predictor.classify_new2(data=predictor.test_data[i], debug=True)))
        print("Test: {}\n".format(predictor.test_answer[i]))
        raw_input()


    # predictor.dump()



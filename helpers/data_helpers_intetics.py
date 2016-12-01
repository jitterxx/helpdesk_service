#!/usr/bin/python3 -t
# coding: utf8


import re
import os
import numpy as np
import nltk


f1 = re.compile('[^абвгджзёеыйиклмнуопрстфчхшщцэюяьъqwertyuiopasdfghjklzxcvbnm ]+')
f2 = re.compile('\s{2,}')
f3 = re.compile(' не ')
f4 = re.compile(' ни ')

def clean_str_new(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    STOP_WORDS_CUSTOM = ["как", "или", "который", "которых", "тот", "около", "они", "для", "это", "при",
                         "кроме", "того", "чем", "под", "них", "его", "лат", "также", "также", "этой", "этого",
                         "com", "вам", "вам", "вами", "вас", "ваше", "все",
                         "добрый", "спасибо", "здравствуйте", 'да', 'мы']

    STOP_WORDS_RUS = nltk.corpus.stopwords.words('russian')
    STOP_WORDS_ENG = nltk.corpus.stopwords.words('english')

    # print('Raw sting: ', string)
    # print(type(string), len(string))

    string = string.lower()

    # string = re.sub('[^абвгджзёеыйиклмнуопрстфчхшщцэюяьъqwertyuiopasdfghjklzxcvbnm ]+', ' ', string)
    string = f1.sub(' ', string)

    # string = re.sub("\s{2,}", " ", string)
    string = f2.sub(' ', string)
    # string = re.sub(' не ', ' не_', string)
    string = f3.sub(' не_', string)
    # string = re.sub(' ни ', ' ни_', string)
    string = f4.sub(' ни_', string)

    string = " ".join([one for one in string.split(' ') if len(one) >= 2 and
                       one not in STOP_WORDS_CUSTOM + STOP_WORDS_RUS + STOP_WORDS_ENG])

    # print('\nClear sting: {}'.format(string))
    # print(type(string), len(string))
    # input()

    return string


def clean_str_orig(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    #print('Raw sting: ', string)
    #print(type(string), len(string))

    string = re.sub(r"[^А-Яа-яA-Za-z0-9(),!?\'\`]", " ", string)
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


    #print('\nClear sting: {}'.format(string))
    #print(type(string), len(string))
    #input()

    return string.strip().lower()


def load_data_and_labels_new_helpers(data_dir=None):

    positive_examples = list()
    negative_examples = list()

    if data_dir:
        data_folder = data_dir
    else:
        data_folder = './data'

    train_data = {'conflict': list(), 'normal': list()}

    for current_cat in list(train_data.keys()):
        print("{}/{}/".format(data_folder, current_cat))
        file_list = list()
        for root, dirs, f_list in os.walk("{}/{}/".format(data_folder, current_cat)):
            for oo in f_list:
                if 'class.nfo' != oo:
                    file_list.append('{}/{}/{}'.format(data_folder, current_cat, oo))

        print("Готовим категорию: {} - {} сообщений".format(current_cat, len(file_list)))

        data = list()
        for ff in file_list:
            f = open(ff, 'r')
            ss = re.split(r'[\r|\n]+', f.read())
            train_data[current_cat].append(" ".join(ss))

    positive_examples = train_data['conflict']
    negative_examples = train_data['normal']

    # Split by words
    x_text = positive_examples + negative_examples
    # print('Old: ', [clean_str2(sent) for sent in x_text])
    # print('New: ', [data_helpers_test.clean_str_new(sent) for sent in x_text])

    x_text = [clean_str_orig(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]

    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    #print x_text
    #print y
    #raw_input()

    return [x_text, y]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



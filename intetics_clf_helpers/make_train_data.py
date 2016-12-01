#!/usr/bin/python3 -t
# coding: utf8


import uuid
import sys
sys.path.extend(['..'])

import argparse
import datetime
import sqlalchemy
import logging
import os
import shutil

import requests
import json

PATH = "{}/{}_train_data".format(sys.argv[1], sys.argv[2])
fname = {'conflict': 'rt-polarity.neg', 'normal': 'rt-polarity.pos'}

for cat in ['conflict', 'normal']:

    CAT_PATH = "{}/{}".format(PATH, cat)

    if os.path.exists(CAT_PATH):
        print("Удаляем старый каталог и файлы в нем - {}".format(CAT_PATH))
        shutil.rmtree(CAT_PATH)

    print("Создаем новый каталог: {}".format(CAT_PATH))
    os.makedirs(CAT_PATH)

    print('Читаем файл {}/{}'.format(PATH, fname[cat]))
    for data in open("{}/{}".format(PATH, fname[cat]), 'r', encoding="latin-1").readlines():
        filename = uuid.uuid4().__str__()
        fnew = open("{}/{}".format(CAT_PATH, filename), "w")
        fnew.write(data)
        fnew.write("\n\n")
        fnew.close()


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


import requests
import json


from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import shutil

Base = automap_base()

# engine, suppose it has two tables 'user' and 'address' set up
sql_uri = "mysql://%s:%s@%s:%s/%s?charset=utf8" % ('root', 'Cthutq123', '127.0.0.1', '3306', 'yurburo')
engine = create_engine(sql_uri)

# reflect the tables
Base.prepare(engine, reflect=True)

# mapped classes are now created with names by default
# matching that of the table name.
TrainData = Base.classes.train_data
UserTrainData = Base.classes.user_train_data
Category = Base.classes.category
TrainAPI = Base.classes.train_api
Msg = Base.classes.email_cleared_data

session = Session(engine)

PATH = "{}/{}_train_data".format(sys.argv[1], sys.argv[2])

cats = session.query(Category.code).all()

for cat, in cats:

    CAT_PATH = "{}/{}".format(PATH, cat)

    if os.path.exists(CAT_PATH):
        print("Удаляем старый каталог и файлы в нем - {}".format(CAT_PATH))
        shutil.rmtree(CAT_PATH)

    print("Создаем новый каталог: {}".format(CAT_PATH))
    os.makedirs(CAT_PATH)

    resp1 = session.query(TrainData).filter(TrainData.category == cat).all()
    resp2 = session.query(UserTrainData).filter(UserTrainData.category == cat).all()
    pool = [resp1, resp2]

    resp3 = session.query(Msg).\
        join(TrainAPI, TrainAPI.message_id == Msg.message_id).\
        filter(TrainAPI.user_answer == cat,
               Msg.channel_type == 0).\
        all()

    pool = [resp3]

    for resp in pool:
        for one in resp:

            if len(one.message_text) > 5:
                data = one.message_title + '\n' + one.message_text
                filename = uuid.uuid4().__str__()
                f = open("{}/{}".format(CAT_PATH, filename), "w")
                f.write(data)
                f.write("\n\n")
                f.close()
            else:
                print("Текст отсутствует!  TICKET ID - {}".format(one.id))



session.close()

#!/usr/bin/python -t
# coding: utf8


import uuid
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

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

resp = session.query(Ticket, Article, Field).\
    join(Article, Article.ticket_id == Ticket.id).\
    join(Field, Field.object_id == Ticket.id).\
    filter(
           Field.field_id == 3).\
    limit(1)

resp = session.query(Field.value_text, sqlalchemy.func.count(Ticket)).\
    join(Ticket, Ticket.id == Field.object_id).\
    filter(
           Field.field_id == 3).\
    group_by(Field.value_text).\
    all()

for one, two in resp:
    resp2 = session.query(Value.value_data).filter(Value.value_text == one).one_or_none()
    if resp2:
        res = resp2[0]
        print('{} - {} - {}'.format(one, res, two))


PATH = "{}/{}_train_data".format(sys.argv[1], sys.argv[2])


resp = session.query(Value).all()
cat_list = list()


for one in resp:
    current_cat = one.value_text
    cat_list.append(current_cat)
    CAT_PATH = "{}/{}".format(PATH, current_cat)

    if os.path.exists(CAT_PATH):
        print("Удаляем старый каталог и файлы в нем - {}".format(CAT_PATH))
        shutil.rmtree(CAT_PATH)

    print("Создаем новый каталог: {}".format(CAT_PATH))
    os.makedirs(CAT_PATH)

    resp2 = session.query(Ticket).\
        join(Field, sqlalchemy.and_(Field.value_text == one.value_text,
                                    Field.field_id == 3)).\
        filter(Ticket.id == Field.object_id).\
        all()

    for ticket in resp2:
        # print('Ищем текст для TICKET ID: {}'.format(ticket.id))
        resp3 = session.query(Article).filter(Article.ticket_id == ticket.id).order_by(Article.id).all()

        if resp3:

            data = resp3[0].a_subject + '\n' + resp3[0].a_body
            # data = resp3[0].a_body

            if len(data) > 5:
                filename = uuid.uuid4().__str__()
                f = file("{}/{}".format(CAT_PATH, filename), "w")
                f.write(data)
                f.write("\n\n")
                f.close()
            else:
                print("Текст отсутствует!  TICKET ID - {}".format(ticket.id))
        else:
            print('Переписки для TICKET ID - {}, не найдено. '.format(ticket.id))



print cat_list


session.close()

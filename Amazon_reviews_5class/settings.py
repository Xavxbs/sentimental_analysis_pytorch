# -*- coding: utf-8 -*-
# @Time : 2019/12/18 19:50
# @Author : Xav
# @File : settings.py

DATA_PATH = './data/reviews_Amazon_Instant_Video_5.json'
LOAD_PATH = './runs/1578797851/best_model.pth'
LOW_THRESHOLD = 0
HIGH_THRESHOLD = 0.8
SEQ_LENGTH = 200
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 50
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
OUTPUT_DIM = 5
EPOCHS = 20
LOAD = False
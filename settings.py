# -*- coding: utf-8 -*-
# @Time : 2019/12/18 19:50
# @Author : Xav
# @File : settings.py

TEST_NEG_PATH = './data/test/neg/'
TEST_POS_PATH = './data/test/pos/'
TRAIN_NEG_PATH = './data/train/neg/'
TRAIN_POS_PATH = './data/train/pos/'
LOW_THRESHOLD = 10
HIGH_THRESHOLD = 0.8
SEQ_LENGTH = 200
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 50
INPUT_DIM = 30764
EMBEDDING_DIM = 400
HIDDEN_DIM = 256
OUTPUT_DIM = 1
EPOCHS = 20
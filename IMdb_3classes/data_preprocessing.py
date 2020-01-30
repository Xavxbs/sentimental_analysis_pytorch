# -*- coding: utf-8 -*-
# @Time : 2019/12/18 19:25
# @Author : Xav
# @File : data_preprocessing.py
import math
import os
from collections import Counter
import numpy as np
from Amazon_reviews_5class import settings
import random
import string

def load_data():
    # 设置文件路径
    train_pos_files = os.listdir(settings.TRAIN_POS_PATH)
    train_neg_files = os.listdir(settings.TRAIN_NEG_PATH)
    test_pos_files = os.listdir(settings.TEST_POS_PATH)
    test_neg_files = os.listdir(settings.TEST_NEG_PATH)
    train_neg = []
    train_pos = []
    test_neg = []
    test_pos = []
    labels = []
    # 从分散的文件中读取数据
    for file in train_neg_files:
        filepath = settings.TRAIN_NEG_PATH + file
        name = file.split(".")[0]
        star =int(name.split("_")[1])-1
        if star < 3:
            labels.append(0)
        elif star > 6:
            labels.append(1)
        else:
            labels.append(2)
        with open(filepath, 'r') as f:
            train_neg.extend(f.readlines())
    for file in train_pos_files:
        filepath = settings.TRAIN_POS_PATH + file
        name = file.split(".")[0]
        star =int(name.split("_")[1])-1
        if star < 3:
            labels.append(0)
        elif star > 6:
            labels.append(1)
        else:
            labels.append(2)
        with open(filepath, 'r') as f:
            train_pos.extend(f.readlines())
    for file in test_neg_files:
        filepath = settings.TEST_NEG_PATH + file
        name = file.split(".")[0]
        star =int(name.split("_")[1])-1
        if star < 3:
            labels.append(0)
        elif star > 6:
            labels.append(1)
        else:
            labels.append(2)
        with open(filepath, 'r') as f:
            test_neg.extend(f.readlines())
    for file in test_pos_files:
        filepath = settings.TEST_POS_PATH + file
        name = file.split(".")[0]
        star =int(name.split("_")[1])-1
        if star < 3:
            labels.append(0)
        elif star > 6:
            labels.append(1)
        else:
            labels.append(2)
        with open(filepath, 'r') as f:
            test_pos.extend(f.readlines())
    # 合并数据
    reviews = train_neg+ train_pos + test_neg  + test_pos
    # labels = [0 if i < (len(train_neg) + len(test_neg)) else 1 for i in
    #           range(len(train_neg) + len(train_pos) + len(test_neg) + len(test_pos))]
    random.seed(10)
    random.shuffle(reviews)
    random.seed(10)
    random.shuffle(labels)
    return reviews, np.array(labels)


def Embedding(reviews):
    reviews = [review.lower() + '\n' for review in reviews]
    reviews = ''.join(reviews)[:-1]
    for c in string.punctuation:
        reviews = reviews.replace(c, '')
    text = reviews.replace('\n', ' ')
    # 删除频率小于10的低频次
    # 过于低频的词会造成一些干扰
    text = text.split()
    word_count = Counter(text)
    text = [word for word in text if word_count[word] >= settings.LOW_THRESHOLD]
    # 删除高频词
    # p(w) = 1 - sqrt(t / f(w))
    # f(w)是当前word的频率，f(w) = count(w) / count(text)，频率是频数除以文本包含的总词数
    t = 1e-3
    word_count = Counter(text)
    total_count = len(text)
    word_freq = {word: (1 - math.sqrt(t / (count / total_count))) for word, count in word_count.items()}
    text = [word for word in text if word_freq[word] < settings.HIGH_THRESHOLD]
    vocab = set(text)
    vocab_list = list(vocab)
    global input_dim
    input_dim = len(vocab_list)
    vocab_list.sort()
    word2int = {word: index for index, word in enumerate(vocab_list)}
    reviews_ind = []
    reviews = [review.split() for review in reviews.split('\n')]
    for review in reviews:
        reviews_ind.append([word2int[c] for c in review if word2int.get(c) is not None])
    return reviews_ind, input_dim


def Padding(reviews_ind):
    data_padded = np.zeros((len(reviews_ind), settings.SEQ_LENGTH), dtype=int)
    for i, row in enumerate(reviews_ind):
        data_padded[i, -len(row):] = np.array(row)[:settings.SEQ_LENGTH]
    print('data padded')
    return data_padded


def split_data(x, y):
    split_idx = int(len(x) * settings.TRAIN_TEST_SPLIT)
    train_x, remaining_x = x[:split_idx], x[split_idx:]
    train_y, remaining_y = y[:split_idx], y[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    print('data splitted')
    return train_x, train_y, test_x, test_y, val_x, val_y


if __name__ == '__main__':
    reviews, labels = load_data()
    rev_ind = Embedding(reviews)
    features = Padding(rev_ind)
    train_x, train_y, test_x, test_y, val_x, val_y = split_data(features, labels)

# -*- coding: utf-8 -*-
# @Time : 2019/12/18 19:25
# @Author : Xav
# @File : data_preprocessing.py
import math
from collections import Counter
import numpy as np
from Amazon_reviews_5class import settings
import random
import string
import pandas as pd


def load_data():
    # 设置文件路径
    data = pd.read_csv(settings.DATA_PATH, encoding='ISO-8859-1', sep=',', header=None)
    reviews = data.iloc[:,5]
    labels_raw = data.iloc[:,0]
    labels = []
    for raw in labels_raw:
        if raw is 0:
            labels.append(0)
        elif raw is 2:
            labels.append(1)
        else:
            labels.append(2)
    random.seed(10)
    reviews = np.array(reviews).tolist()
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
    # 这里没有删除，因为处理的是Twitter，有很多短句子
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

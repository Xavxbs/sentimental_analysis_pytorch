# -*- coding: utf-8 -*-
# @Time : 2019/12/18 20:55
# @Author : Xav
# @File : train.py
from data_preprocessing import *
from settings import *
from model import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
import numpy as np

def train_epoch_progress(model, train_iter, loss_function, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for inputs, labels in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        truth_res += list(labels)
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        pred = model(inputs)
        pred_label = pred.data.numpy()
        pred_res += [1 if x > 0.5 else 0 for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred.float(), labels.float())
        avg_loss += loss.data
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for inputs, labels in data:
        truth_res += list(labels)
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        pred = model(inputs)
        pred_label = pred.data.numpy()
        pred_res += [1 if x > 0.5 else 0 for x in pred_label]
        loss = loss_function(pred.float(), labels.float())
        avg_loss += loss.data
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc



USE_GPU = torch.cuda.is_available()
timestamp = str(int(time.time()))
best_dev_acc = 0.0

reviews, labels = load_data()
rev_ind = Embedding(reviews)
features = Padding(rev_ind)
train_x, train_y, test_x, test_y, val_x, val_y = split_data(features, labels)
# 准备数据
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)
# 生成网络
model = LSTMSentiment(EMBEDDING_DIM,HIDDEN_DIM,INPUT_DIM,OUTPUT_DIM, USE_GPU,BATCH_SIZE)

if USE_GPU:
    model = model.cuda()


best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.BCELoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_loader, loss_function, optimizer, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    dev_acc = evaluate(model, valid_loader, loss_function, 'Dev')

    # 保存最佳的网络&测试模型
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm '+ out_dir + '/best_model' + '.pth')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        test_acc = evaluate(best_model, test_loader, loss_function, 'Test')
test_acc = evaluate(best_model, test_loader, loss_function, 'Final Test')
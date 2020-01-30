# -*- coding: utf-8 -*-
# @Time : 2019/12/18 20:55
# @Author : Xav
# @File : train.py
from Amazon_reviews_5class.data_preprocessing import *
from Amazon_reviews_5class.settings import *
from Amazon_reviews_5class.model import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import time
import os
from tqdm import tqdm


def train_epoch_progress(model, train_iter, loss_function, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    correct = 0
    for inputs, labels in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        truth_res += list(labels)
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        if USE_GPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        pred = model(inputs)
        # pred_label = pred.data.numpy()
        # pred_res += [1 if x > 0.5 else 0 for x in pred_label]
        pred_res += pred.max(1, keepdim=True)[1].view(1,-1).tolist()[0]
        model.zero_grad()
        #loss = loss_function(pred.float(), labels.float())
        loss = loss_function(pred.float(), labels)
        avg_loss += loss.data
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    # auc = roc_auc_score(truth_res,pred_res)
    # return avg_loss, acc , auc
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
        if USE_GPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        pred = model(inputs)
        # pred_label = pred.data.numpy()
        # pred_res += [1 if x > 0.5 else 0 for x in pred_label]
        pred_res += pred.max(1, keepdim=True)[1].view(1, -1).tolist()[0]
        #loss = loss_function(pred.float(), labels.float())
        loss = loss_function(pred.float(), labels)
        avg_loss += loss.data
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    #auc_score = roc_auc_score(truth_res,pred_res)
    # print(name + ': loss %.2f acc %.1f auc %.3f' % (avg_loss, acc*100, auc_score))
    # return acc, auc_score
    print(name + ': loss %.2f acc %.3f' % (avg_loss, acc))
    return acc



USE_GPU = torch.cuda.is_available()
timestamp = str(int(time.time()))
best_dev_acc = 0.0

reviews, labels = load_data()
rev_ind, input_dim = Embedding(reviews)
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
model = LSTMSentiment(EMBEDDING_DIM,HIDDEN_DIM,input_dim,OUTPUT_DIM, USE_GPU,BATCH_SIZE)

if USE_GPU:
     model = model.cuda()


best_model = model
if LOAD:
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()
    print('Loading...')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#loss_function = nn.BCELoss()
loss_function = nn.NLLLoss()
print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    # avg_loss, acc, auc = train_epoch_progress(model, train_loader, loss_function, optimizer, epoch)
    # tqdm.write('Train: loss %.2f acc %.1f auc %.3f' % (avg_loss, acc*100, auc))
    avg_loss, acc = train_epoch_progress(model, train_loader, loss_function, optimizer, epoch)
    tqdm.write('Train: loss %.2f acc %.3f' % (avg_loss, acc))
    # dev_acc, dev_auc = evaluate(model, valid_loader, loss_function, 'Dev')
    dev_acc = evaluate(model, valid_loader, loss_function, 'Dev')

    # 保存最佳的网络&测试模型
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm '+ out_dir + '/best_model' + '.pth')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        # test_acc, test_auc = evaluate(best_model, test_loader, loss_function, 'Test')
        test_acc = evaluate(best_model, test_loader, loss_function, 'Test')
test_acc = evaluate(best_model, test_loader, loss_function, 'Final Test')
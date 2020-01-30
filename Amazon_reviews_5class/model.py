# -*- coding: utf-8 -*-
# @Time : 2019/12/18 23:29
# @Author : Xav
# @File : model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMSentiment(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,batch_first=True)
        self.dense = nn.Linear(hidden_dim, label_size)
        #self.sig = nn.Sigmoid()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        y = self.dense(lstm_out)
        # sig_out = self.sig(y)
        # sig_out = sig_out.view(x.size(0), -1)
        # sig_out = sig_out[:, -1]
        # return sig_out
        return F.log_softmax(y, dim=1).view(x.size(0),-1)[:, -5:]

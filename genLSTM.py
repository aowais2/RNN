#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:22:24 2019

@author: anomander
"""
import numpy as np
import torch
import torch.nn as nn

class dataset:
    def __init__(self,input):
        self.inp = input
        self.targ = np.roll(self.inp,1,axis=1)
    def numtotorch(self,input):
        self.input = torch.tensor(self.inp,dtype=torch.float)
        self.target = torch.tensor(self.targ,dtype=torch.float)
        return self.input,self.target

class LSTMmodule(nn.Module):
    def __init__(self,n_in,n_hid,n_out,batch_size,n_layers):
        super(LSTMmodule,self).__init__()
        self.n_inputs = n_in
        self.n_hidden = n_hid
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_output = n_out
        
        self.lstm = nn.LSTM(self.n_inputs,self.n_hidden,self.n_layers)
        self.linear = nn.Linear(self.n_hidden,self.n_output)
        
    def init_hidden(self):
        return (torch.zeros(self.n_layers,self.batch_size,self.n_hidden),
                torch.zeros(self.n_layers,self.batch_size,self.n_hidden))
        
    def forward(self,inp):
        self.lstm_out,self.hidden = self.lstm(inp,self.init_hidden())
        self.ypred = self.linear(self.lstm_out)
        return self.ypred
    
a = np.load('./data/dataset/motor_1_30_16.npy')        
data = dataset(a)
d = data.numtotorch(a)
n_in,n_hid,n_out,batch_size,n_layers,n_epochs = a.shape[2],20,a.shape[2],a.shape[1],2,12000    
model = LSTMmodule(n_in,n_hid,n_out,batch_size,n_layers)
lossfn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
hist = np.zeros(n_epochs)

for t in range(n_epochs):
    model.zero_grad()
    #model.hidden = model.init_hidden()
    y_pred = model(d[0])
    
    loss = lossfn(y_pred,d[1])
    if t%100==0:
        print("Epoch: ",t," MSE ",loss.item())
    hist[t] = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
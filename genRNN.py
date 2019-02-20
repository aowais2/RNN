#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:40:19 2019
RNN using PyTorch
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

class RNNmodule(nn.Module):
    def __init__(self,n_inputs,n_hidden,n_layers,batch_size,n_output):
        super(RNNmodule,self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_output = n_output
        
        self.rnn = nn.RNN(self.n_inputs,self.n_hidden,self.n_layers)
        #num_layer*batch*hidden size
        self.linear = nn.Linear(self.n_hidden,self.n_output)
        
    def forward(self,inputs):
        self.out, self.hx = self.rnn(inputs,torch.randn([self.n_layers,
                                            self.batch_size,self.n_hidden]))
        self.y_pred = self.linear(self.out)
        return self.y_pred
        
a = np.load('./data/dataset/motor_1_30_16.npy')        
data = dataset(a)
d = data.numtotorch(a)

loss_fn = nn.MSELoss()
learn,n_in,n_hid,n_layers,batch_size,n_out,n_epoch = 1e-3,16,20,2,a.shape[1],16,10000
model = RNNmodule(n_in,n_hid,n_layers,batch_size,n_out)
optim = torch.optim.Adam(model.parameters(),lr = learn)

for t in range(n_epoch):
    
    model.zero_grad()
    ypred = model(d[0])
    loss = loss_fn(ypred,d[1])
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os
import numpy as np
import Data
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import argparse
import torch.nn.functional as F

from tqdm import tqdm, trange

import Data
from torch.utils.data import DataLoader
import argparse

import warnings
warnings.filterwarnings("ignore")

class AccentModel(torch.nn.Module):
    
    def __init__(self, blockSize,input_dim, hidden_dim, batch_size, output_dim=17, num_layers=4):
        super(AccentModel, self).__init__()
        
        
        self.blockSize = blockSize
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers        
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True)
        
        self.linear = nn.Linear(self.hidden_dim*2, output_dim)        
        
        self.atten = nn.Linear(self.hidden_dim*2,1)
        self.initParams()
        
    def initParams(self):
        for param in self.parameters():
            if len(param.shape)>1:
                torch.nn.init.xavier_normal_(param)
                

    def init_hidden(self):
        
        return (torch.zeros(1,1,self.output_dim),torch.zeros(1,1,self.output_dim))
    
    def attention(self,decoder_hidden,lstm_outs):
        weights = []
        
        for i in range(len(lstm_outs)):
            weights.append(self.atten(lstm_outs[i]))
        
        normalized_weights = F.softmax(torch.cat(weights,1),1)
        batchSize = lstm_outs.shape[1]
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 lstm_outs.view(batchSize,-1,self.hidden_dim*2))
                    
        return attn_applied

    
    def temporal(self, x):
        
        seq = x.shape[1]
        h_1,c_1 = self.lstmCell(x[:,0,:])
        for i in range(1,seq):
            h_1,c_1 = self.lstmCell(x[:,i,:],(h_1,c_1))
            #if i%30 ==0:
                
                #h_1 = h_1.detach()
                #c_1 = c_1.detach()
        
        return h_1
        

    def forward(self, x):
                
        x = x.transpose(0,1)
        lstm_out, (hidden,cell) = self.lstm(x)
        att_out = self.attention(self.init_hidden, lstm_out)
        logits = self.linear(att_out)

        o = F.softmax(logits, dim=2)
                        
        return o

#======================================================================================  

def validate(model,dataloader):
    model.eval()
    lossT = 0
    criterion = nn.NLLLoss()#nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 1
    with torch.no_grad():
        for sample in dataloader:
            try:
                
                wavX, t_accent = sample
                specWav = wavX.to(device)
                t_accent = t_accent.to(device)
                
                specWavT = torch.transpose(specWav,1,2)
                
                pred = model.forward(specWavT)
                count+=1
                
                loss = criterion(pred.squeeze(1),t_accent)
                lossT+=loss.item()
               
            except Exception as e:
                #print('Skipping category', t_accent)
                print(str(count)+' out of '+str(len(dataloader)))
                #print('pred dim', pred.shape)
                print('accent ',t_accent.shape)

                print(e)                     
                continue            
            
            
    model.train()       
    return lossT/count
    

def train(nepochs,trainDataloader,valDataloader,learningRate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentModel(blockSize=201,input_dim = 201, hidden_dim=100, batch_size=30, output_dim=17, num_layers=2)
    model.to(device) 
    
    model = nn.DataParallel(model)
    
    #initialize the optimizer for paramters
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    model.train(mode=True)    
    
    criterion = nn.CrossEntropyLoss()
    minValLoss = 10000000000000
    load = False
    
    if load and (os.path.isfile('AttentionAccentModel_feedForward_best.pt')):
        print('..... loading model')
        checkpoint = torch.load('AttentionAccentModel_feedForward_last.pt')
        minValLoss = checkpoint['minValLoss']
        model.load_state_dict(checkpoint['state_dict'])        
    lr = learningRate
    
    for epoch in range(nepochs):
        iterator = iter(trainDataloader)
        totalLoss = 0
        total_len = 0
        correct = 0
        with trange(len(trainDataloader)) as t:
            
            for index in t:
                try:
                    if epoch>5:
                        lr = learningRate*np.exp(-0.0001*epoch)
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
  
                    sample = next(iterator)
                    
                    wavX, t_accent = sample
                    
                    specWav = wavX.to(device)
                    t_accent = t_accent.to(device)
                    
                    specWavT = torch.transpose(specWav,1,2)
                    optimizer.zero_grad()
                    pred_a = model.forward(specWavT)
                    
                    _, predicted = torch.max(pred_a.data, 2)
                    total_len += t_accent.size(0)
                    correct += (predicted[0]==t_accent).sum().item()
                    acc = 100*(correct/total_len)
                    
                    loss = criterion(pred_a.squeeze(1),t_accent)
                    
                    totalLoss = totalLoss+loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    
                    t.set_description(f"epoc : {epoch}, loss {totalLoss/(index+1)}, lr: {lr}, accuracy: {acc}")
                        
                except Exception as e:
                    print(e)                     
                    continue
            checkpoint = {
                'state_dict': model.state_dict(),
                'minValLoss': totalLoss/(index+1)
            }
            # save the last checkpoint
            torch.save(checkpoint, 'AttentionAccentModel_feedForward_last.pt')   
            
            validLoss = validate(model,valDataloader)
            print('validation loss ',validLoss)
            if validLoss < minValLoss:
                
                minValLoss = validLoss
                
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'minValLoss': minValLoss,
                }
                torch.save(checkpoint, 'AttentionAccentModel_feedForward_best.pt')                
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Accent Model arguments')
    parser.add_argument("--clippath", type=str, default = 'data/clips/')
    parser.add_argument("--meta", type=str, default = 'data/')
    
    args = parser.parse_args()
    Path_Wav = args.clippath
    Path_Meta = args.meta    
    
    trainDataset = Data.VoiceData(Path_Wav,Path_Meta+'train.tsv')
    trainDataloader = DataLoader(trainDataset,batch_size=30,shuffle=True, num_workers=0,collate_fn=Data.collate_fn)    
    
    valDataset = Data.VoiceData(Path_Wav,Path_Meta+'dev.tsv')
    valDataloader = DataLoader(valDataset,batch_size=30,shuffle=True, num_workers=0,collate_fn=Data.collate_fn) 
    
    #testDataset = Data.VoiceData(Path_Wav,Path_Meta+'test.tsv')
    #testDataloader = DataLoader(testDataset,batch_size=30,shuffle=True, num_workers=0,collate_fn=Data.collate_fn) 

    train(nepochs=50000, trainDataloader=trainDataloader, valDataloader=valDataloader, 
          learningRate=0.1)        

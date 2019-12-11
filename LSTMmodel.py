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

#import matplotlib.pyplot as plt
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
        self.num_layers = num_layers        
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,bidirectional = True)
        
        self.lstmCell = nn.LSTMCell(self.input_dim,self.hidden_dim)
        
        self.linear = nn.Linear(self.hidden_dim*2, output_dim)
        
        ###################################
        self.fc1 = nn.Linear(self.hidden_dim,500)
        self.fc2 = nn.Linear(500,300)
        
        self.h1 = nn.Linear(300,300)
        self.h2 = nn.Linear(300,300)
        self.h3 = nn.Linear(300,300)
        
        self.fc3 = nn.Linear(300,100)
        self.fc4 = nn.Linear(100,output_dim)
    
        ###################################
    
        self.initParams()
        
    def initParams(self):
        for param in self.parameters():
            if len(param.shape)>1:
                torch.nn.init.xavier_normal_(param)
                
    
    def encode(self, x):

        #x = x.view(x.shape[0],-1)
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, self.blockSize)
        
        
        x = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(x))


        return h
    
    def decode(self, h):

        h = F.leaky_relu(self.fc3(h))
        o = F.softmax(self.fc4(h))

        return o
    
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
        
        #lstm_out = self.temporal(x)
        
        
        #x = x[torch.arange(x.size(0)), x.shape[1] - 1]
        #lstm_out, hidden = self.lstm(x)
        #x = F.leaky_relu(self.fc1(lstm_out))
        #h = F.leaky_relu(self.fc2(x))
        
        #h = F.leaky_relu(self.h1(h)) 
        #h = F.leaky_relu(self.h2(h)) 
        #h = F.leaky_relu(self.h3(h)) 
        
        #h = F.leaky_relu(self.fc3(h))
        #logits = self.fc4(h)        
        
        x = x.transpose(0,1)
        lstm_out,(hidden,cell) = self.lstm(x)
        # taking both directions from the last layer of lstm
        batchSize = x.shape[1]
        hidden_l = hidden.view(self.num_layers,2,batchSize,self.hidden_dim)
        hidden_l = hidden_l[-1].transpose(0,1)
        hidden_cat = hidden_l.reshape(batchSize,self.hidden_dim*2)
        
        
        logits = self.linear(hidden_cat)
        
        #o = F.softmax(logits, dim=1)
        #glue the encoder and the decoder together
        #h = self.encode(x)
        #o = self.decode(h)
                        
        return logits, hidden_cat
    
    def getLastLayer(self,x):
        
        _,layer = self.forward(x)
        
        return layer

#==================================================================================  

def validate(model,dataloader):
    model.eval()
    lossT = 0
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 1
    
    with torch.no_grad():
        for sample in dataloader:
            try:
                
                wavX, t_accent = sample
                specWav = wavX.to(device)
                t_accent = t_accent.to(device)
                
                specWavT = torch.transpose(specWav,1,2)
                
                pred,_ = model.forward(specWavT)
                
                
                ##### debugging #################
                count+=1

                #################################
                #print('accent label ', t_accent)
                #print(str(count)+' out of '+str(len(dataloader))
                
                loss = criterion(pred,t_accent)
                lossT+=loss.item()
                
                #if count>100:
                    
                    #break                
            except Exception as e:
                #print('Skipping category', t_accent)
                print(str(count)+' out of '+str(len(dataloader)))
                print('pred dim', pred.shape)
                print('accent ',t_accent.shape)

                print(e)                     
                continue            
            
            
    model.train()       
    return lossT/count
    

def train(nepochs,trainDataloader,valDataloader,learningRate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AccentModel(blockSize=201,input_dim = 201, hidden_dim=100, batch_size=8, output_dim=17, num_layers=4)
    model.to(device) 
    #initialize the optimizer for paramters
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    model.train(mode=True)    
    model= nn.DataParallel(model)    
    
    criterion = nn.CrossEntropyLoss()
    minValLoss = 10000000000000
    load = True
    if load and (os.path.isfile('AccentModel_feedForward_best.pt')):
        print('..... loading model')
        checkpoint = torch.load('AccentModel_feedForward_last.pt')
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
                        print('kearning rate ', lr)
                        
                        
                    sample = next(iterator)
                    
                    wavX, t_accent = sample
                    
                    specWav = wavX.to(device)
                    t_accent = t_accent.to(device)
                    
                    specWavT = torch.transpose(specWav,1,2)
                    optimizer.zero_grad()
                    pred_a,_ = model.forward(specWavT)
                    
                    _,predicted = torch.max(F.softmax(pred_a,1).data,1)
                    total_len += t_accent.size(0)
                    correct+= (predicted==t_accent).sum().item()
                    acc = 100*(correct/total_len)
                    
                    loss = criterion(pred_a,t_accent)
                    
                    totalLoss = totalLoss+loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    
                    t.set_description(f"epoc : {epoch}, loss {totalLoss/(index+1)}, lr: {lr}, accuracy: {acc}")
                    #if index>100:
                        #break
                        
                except Exception as e:
                    print(e)                     
                    continue
            checkpoint = {
                'state_dict': model.state_dict(),
                'minValLoss': totalLoss/(index+1)
            }
            # save the last checkpoint
            torch.save(checkpoint, 'AccentModel_feedForward_last.pt')   
            
            validLoss = validate(model,valDataloader)
            print('validation loss ',validLoss)
            if validLoss < minValLoss:
                
                minValLoss = validLoss
                
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'minValLoss': minValLoss,
                }
                torch.save(checkpoint, 'AccentModel_feedForward_best.pt')                
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Accent Model arguments')
    parser.add_argument("--clippath", type=str, default = 'data/clips/')
    parser.add_argument("--meta", type=str, default = 'data/')
    
    args = parser.parse_args()
    Path_Wav = args.clippath
    Path_Meta = args.meta    
    
    trainDataset = Data.VoiceData(Path_Wav,Path_Meta+'train.tsv')
    trainDataloader = DataLoader(trainDataset,batch_size=30,shuffle=True, num_workers=6,collate_fn=Data.collate_fn)    
    
    valDataset = Data.VoiceData(Path_Wav,Path_Meta+'dev.tsv')
    valDataloader = DataLoader(valDataset,batch_size=30,shuffle=True, num_workers=6,collate_fn=Data.collate_fn) 
    
    #testDataset = Data.VoiceData(Path_Wav,Path_Meta+'test.tsv')
    #testDataloader = DataLoader(testDataset,batch_size=1,shuffle=True, num_workers=0,collate_fn=Data.collate_fn) 

    train(nepochs=50000, trainDataloader=trainDataloader, valDataloader=valDataloader, 
          learningRate=0.001)        

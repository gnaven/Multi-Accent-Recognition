from __future__ import print_function, division, absolute_import, unicode_literals
import LSTMmodel as network
import os
import numpy as np
import Data
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import argparse
import torch.nn.functional as F
import Data
from torch.utils.data import DataLoader
import argparse
import csv
import traceback
import warnings
warnings.filterwarnings("ignore")

from sklearn import cluster

from sklearn.preprocessing import normalize

import pandas as pd
def createDataset(fname,data,label,writer):
    
    for i in range(data.shape[0]):
        feats = data[i].tolist()
        y = label[i].tolist()
        writer.writerow(feats+[y])      
        
    pass
    
def runModel(modelName,dataloader,fname):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = network.AccentModel(blockSize=201,input_dim = 201, hidden_dim=100, batch_size=8, output_dim=17, num_layers=4)
    model= nn.DataParallel(model)    
    
    if os.path.isfile(modelName):
        print('..... loading model')
        checkpoint = torch.load(modelName,map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])         
        model.to(device)  
        
    else:
        print('model was not found.....')
    lossT = 0
    count = 0
    total_len = 0
    criterion = nn.CrossEntropyLoss()   
    correct = 0
    acc = 0
    with torch.no_grad():
        with open(fname,mode ='w') as csv_file:
            writer = csv.writer(csv_file,delimiter= ',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
            feat_names = ['feat'+str(i) for i in range(200)] + ['Accent']
            writer.writerow(feat_names)                  
            for sample in dataloader:
                try:
                    wavX, t_accent = sample
                    specWav = wavX.to(device)
                    t_accent = t_accent.to(device)
    
                    specWavT = torch.transpose(specWav,1,2)
    
                    pred,lastlayer = model.forward(specWavT)
                    
                    createDataset(fname,lastlayer,t_accent,writer)
                    
                    _,predicted = torch.max(F.softmax(pred,1).data,1)
                    total_len += t_accent.size(0)
                    correct+= (predicted==t_accent).sum().item()
                    acc = 100*(correct/total_len)
                    
                    count+=1
                    loss = criterion(pred,t_accent)
                    lossT+=loss.item()
                    print('running metrics, loss: ',lossT/count, ' accuracy: ', acc,' completed' ,count,'/',len(dataloader))  
                
                except Exception as e:
                    #print('Skipping category', t_accent)
                    print(str(count)+' out of '+str(len(dataloader)))
                    #print('pred dim', pred.shape)
                    #print('accent ',t_accent.shape)
    
                    print(e)   
                    traceback.print_exc()
                    continue            
      
    return (lossT/count), acc

def Clustering(data,k=17):
    df = pd.read_csv(data)
    X = df.drop(['Accent'],axis=1)
    X = normalize(X)
    k_means = cluster.KMeans(n_clusters=k, max_iter=1000, n_jobs=-1)
    y = k_means.fit(X)
    
    df['Cluster'] = y.labels_
    print('summary stats....')
    print('label proportion ', df['Accent'].value_counts(normalize=True)*100)
    print('cluster proportion ', df['Cluster'].value_counts(normalize=True)*100)
    
    return df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Accent Model arguments')
    parser.add_argument("--clippath", type=str, default = 'data/clips/')
    parser.add_argument("--meta", type=str, default = 'data/')
    parser.add_argument("--dataset", type=str, default = 'test')    

    args = parser.parse_args()
    Path_Wav = args.clippath
    Path_Meta = args.meta   
    dataset = args.dataset
    
    if dataset == 'train':
        Dataset = Data.VoiceData(Path_Wav,Path_Meta+'train.tsv')
        Dataloader = DataLoader(Dataset,batch_size=2,shuffle=True, num_workers=0,collate_fn=Data.collate_fn)    
    elif dataset == 'validation':
        Dataset = Data.VoiceData(Path_Wav,Path_Meta+'dev.tsv')
        Dataloader = DataLoader(Dataset,batch_size=30,shuffle=True, num_workers=6,collate_fn=Data.collate_fn) 
    else:     
        Dataset = Data.VoiceData(Path_Wav,Path_Meta+'test.tsv')
        Dataloader = DataLoader(Dataset,batch_size=30,shuffle=True, num_workers=6,collate_fn=Data.collate_fn)    
    
    loss,acc = runModel(modelName='AccentModel_LSTM_best.pt', dataloader = Dataloader, fname='DNNLayer_data_'+dataset+'.csv')
    print('final loss ', loss, ' final accuracy', acc)
    cluster_data = Clustering(data='DNNLayer_data_'+dataset+'.csv',k = 17)
    cluster_data.to_csv('Cluster_DNNLayer_data_'+dataset+'.csv')
    print('........ final csv printed')
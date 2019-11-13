from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import util
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm, trange



class VoiceData(Dataset):
    
    def __init__(self,wavPath,metadataPath):
        
        """
        Initializes the dataset instance taking in the path to directory for 
        waveform files and tsv for metada
        Initializes the encoders to convert categorical data into numerical data

        Below is a representation of the directory for Mozilla Voice data
        Mozilla Voice
        |_clips/
        |_fileinfo
         |_filename
         |_Text
         |_Country
         |_Age
        """    
        self.wavPath = wavPath
        self.metadataPath = metadataPath
        self.samples = []
        
        self.gender_codec = LabelEncoder()
        self.accent_codec = LabelEncoder()
        self.__init__dataset()
        
    def __init__dataset(self):
        metaDF = pd.read_csv(self.metadataPath,sep='\t')
        """
        path
        sentence
        up_votes
        down_votes
        age
        gender
        accent
        """
        usecols = ['path','sentence','age','gender','accent']
        df = metaDF[usecols]
        self.gender_codec.fit(df['gender'].values)
        self.accent_codec.fit(df['accent'].values)
        
        self.samples = df.values
        
    def __len__(self):
        """
        returns the length of the dataset
        """
        return (len(self.samples))
        
        
    def __getitem__(self, index):
        """
        given a particular index it will be able to return a 
        sample from the dataset
        """
        wavX, Fs = util.wavread(self.wavPath+self.samples[index][0])
        t_gender,t_accent = self.one_hot_sample(self.samples[index][3], self.samples[index][4])
        t_age = torch.tensor(normalize(self.samples[index][2]))
        
        return wavX, Fs, t_age, t_gender,t_accent
    
    def one_hot_encode(self,codec,value):
        value_index = codec.transform(value)
        t_value = torch.eye(len(codec.classes_))[value_index]
        return t_value
        
    def one_hot_sample(self,gender,accent):
        
        t_gender = self.one_hot_encode(self.gender_codec,[gender])
        t_ccent = self.one_hot_encode(self.accent_codec,[accent])
        
        return t_gender,t_ccent
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test DataLoader')
    parser.add_argument("--clippath", type=str, default = 'data/clip/')
    parser.add_argument("--meta", type=str, default = 'data/')
    
    args = parser.parse_args()
    Path_Wav = args.clippath
    Path_Meta = args.meta
    dataset = VoiceData(Path_Wav,Path_Meta+'test.tsv')
    
    dataloaderTest = DataLoader(dataset,batch_size=1,shuffle=True, num_workers=2)
    iterator = iter(dataloaderTest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with trange(len(dataloaderTest)) as t:
        
        for idx in t:
            
            sample =next(iterator)
            
            wavX, Fs, t_age, t_gender,t_accent = sample.to(device) 
            print(t_age)
            print(t_gender)
            
            if idx >2:
                break
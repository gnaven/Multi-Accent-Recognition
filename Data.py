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
        usecols = ['path','sentence','gender','accent']
        df = metaDF[usecols]
        df = df.dropna()
        self.gender_codec.fit(list(df['gender'].values))
        
        # only set amount of accents that can be mapped to in the dataset
        
        accent = {'us':0, 'england':1, 'hongkong':2, 'indian':3, 'african':4, 'australia':5,
       'newzealand':6, 'canada':7, 'scotland':8, 'ireland':9, 'philippines':10,
       'wales':11, 'singapore':12, 'malaysia':13, 'other':14,'bermuda':15,'southatlandtic':16}
        
        df['accent'] = df['accent'].apply(lambda x: accent[x])
        #self.accent_codec.fit(accent)
        
        self.samples = df.dropna().values
        
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
        blockSize = 400
        hopSize = 200
        
        wavX, Fs = util.wavread(self.wavPath+self.samples[index][0])
        t_gender = self.one_hot_sample(self.samples[index][2], self.samples[index][3])
        #t_age = torch.tensor(normalize(self.samples[index][2]))
        specWav = util.stft_real(wavX,blockSize=blockSize,hopSize=hopSize)

        t_accent = self.samples[index][3]
        
        
        return specWav, t_accent
    
    def one_hot_encode(self,codec,value):
        value_index = codec.transform(value)
        t_value = torch.eye(len(codec.classes_))[value_index]
        
        return t_value
        
    def one_hot_sample(self,gender,accent):
        
        t_gender = self.one_hot_encode(self.gender_codec,[gender])
        
        #t_accent = self.one_hot_encode(self.accent_codec,[accent])
        
        return t_gender
    
def collate_fn(batch):
    nBatch = len(batch)
    
    spec = batch[0][0]
    maxLen = spec.shape[1]
    nWindow = spec.shape[0]
    for i in range(nBatch):
        spec = batch[i][0]
        maxLen = max(spec.shape[1], maxLen)

    tmp = torch.zeros(nBatch, nWindow, maxLen)
    tmpLabel = torch.zeros(nBatch)
    for i in range(nBatch):
        spec = batch[i][0]
        label = batch[i][1]
        tmp[i, :, :spec.shape[1]] = torch.from_numpy(spec)
        tmpLabel[i] = torch.tensor(int(label))
    
        
    return tmp,tmpLabel.long()

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test DataLoader')
    parser.add_argument("--clippath", type=str, default = 'data/clips/')
    parser.add_argument("--meta", type=str, default = 'data/')
    
    args = parser.parse_args()
    Path_Wav = args.clippath
    Path_Meta = args.meta
    dataset = VoiceData(Path_Wav,Path_Meta+'train.tsv')
    
    dataloaderTest = DataLoader(dataset,batch_size=2,shuffle=True, num_workers=0, collate_fn=collate_fn)
    iterator = iter(dataloaderTest)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with trange(len(dataloaderTest)) as t:
        
        for idx in t:
            
            sample =next(iterator)
            
            wavX, t_accent = sample
            
            wavX=wavX.to(device)
      
            t_accent = t_accent.to(device)
                       
            print(t_accent)
            print(wavX.shape)
            #if idx >2:
                #break
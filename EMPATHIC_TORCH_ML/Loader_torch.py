
import uproot

import glob 
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch

class TorchDataset(Dataset):
    def __init__(self, images, aux_data, labels):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)      # Shape: [N, 1, 24, 24]
        self.aux_data = torch.tensor(aux_data, dtype=torch.float32) # Shape: [N, 1]
        self.labels = torch.tensor(labels, dtype=torch.float32)        # Shape: [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.aux_data[idx], self.labels[idx]
        

class Loader():

    def __init__(self,batch_size, test_split):
    
        self.batch_size = batch_size
        self.test_split = test_split

    def LoaderTorchSets(self,path):
        c = 0
        for file in glob.glob(f"{path}/*.pt"):

            temp = torch.load(file, weights_only=False).dataset
            dataset = WrapperDataset(temp)
            
            if(c == 0):
                concat_set = ConcatDataset([dataset])
            else:
                concat_set = ConcatDataset([concat_set, dataset])
            c+=1
        train_dataset, test_dataset = torch.utils.data.random_split(concat_set, [1-self.test_split, self.test_split])
        train_loader = DataLoader(train_dataset, self.batch_size)
        test_loader = DataLoader(test_dataset, self.batch_size)
        return  train_loader, test_loader


    def Rebatch(data_set, new_batch_size):
        loader = DataLoader(data_set.dataset, batch_size=new_batch_size, shuffle=True)
        return loader



class WrapperDataset(Dataset):
    
    def __init__(self,dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mom, pdg = self.dataset[idx]
        if(int(pdg) == 211):onehot = torch.tensor([1.,0,0])
        elif(int(pdg) == 321):onehot = torch.tensor([0,1.,0])
        else: onehot = torch.tensor([0,0,1.])
        
        return torch.tensor(image), torch.tensor(mom.unsqueeze(-1)), onehot

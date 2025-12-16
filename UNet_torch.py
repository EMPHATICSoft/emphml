import numpy as np
import glob
import torch


from Loader_torch import Loader, TorchDataset, WrapperDataset
from Trainer_torch import Trainer

from Model_torch import Model
#from Plots import *


from argparse import ArgumentParser, RawTextHelpFormatter


import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = ArgumentParser(formatter_class= RawTextHelpFormatter)

parser.add_argument('-input_files_path', '-i', help = "Path to input files with TRB3 hits and momenta reconstruction")
parser.add_argument('-mod', '-m' , choices=['Test', 'Train', 'GetWeights'],
                    help = "Mode of the code:  \n Test: loads a model already trained \n Train: makes a model and trains it \n GetWeights: get the weights only of a trained model\n",
                    required = True)
                    
parser.add_argument('-model_path','-p', help = "path to saved model", required = False)

parser.add_argument('-n_epochs','-n', help = "number of epochs", required = False)

parser.add_argument('-batch','-b', help = "batch size", required = False)

parser.add_argument('-saved_sets', help = "path of saved sets ", required = False)

parser.add_argument('-rebatch', help = "new batch size", required = False)

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    if(args.batch is not None):
        loader = Loader(args.batch, 0.2)
    else:
        loader = Loader(64, 0.2)
    

    if(args.saved_sets is not None):
        print("Loading pre saved sets")
        train_loader, test_loader = loader.LoaderTorchSets(args.saved_sets)

        print(f"Loaded {len(train_loader.dataset), len(test_loader.dataset)} training/testing")
        
        if(args.rebatch is not None):
            train_loader = Loader.Rebatch(train_loader,int(args.rebatch))
            test_loader = Loader.Rebatch(test_loader,int(args.rebatch))
        
    
        
    if(args.mod == "Train"):
    
        if(args.n_epochs is not None):
            print(f"Selected Train mode, n epochs = {args.n_epochs}")
            trainer = Trainer(train_loader, test_loader, n_epoch=int(args.n_epochs))
        else:
            print("Selected Train mode, no number of epochs provided, using default value 10")
            trainer = Trainer(train_loader, test_loader, n_epoch=10)

        trainer.train()
        trainer.test(trainer.model)
        trainer.GetPlots()
        
        trainer.save_model()
        
    elif(args.mod == "Test"):
        
        trainer = Trainer(train_loader, test_loader, n_epoch=0)
        
        model = torch.load(args.model_path, weights_only=False)
        trainer.test(model)
        trainer.save_model()
    
    elif(args.mod == "GetWeights"):  
        model = torch.load(args.model_path, weights_only=False)
        torch.save(model.state_dict(),"./model_weights.pt")

    
    
    
    

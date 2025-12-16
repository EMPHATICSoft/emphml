import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from Plots import ByParticleByMomenta,ConfusionPlot

from Model_torch import Model

particle_names = ["Pion", "Kaon", "Proton"]
class Trainer():

    def __init__(self,train_set, test_set,n_epoch):

        self.uniques = []
        
        self.labels = []

        self.all_momenta = []
        self.all_pred = []
        self.all_labels =[]
        
        self.epochs = n_epoch
        
        self.train_set =  train_set
        self.test_set =  test_set
        
        self.criterion = nn.CrossEntropyLoss()
        
        
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
     
    def train(self):
      
        for epoch in range(self.epochs):
           
            start = time.time()
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (x1_batch, x2_batch, y_batch) in enumerate(self.train_set):
               
                print(f"  Batch {batch_idx+1}/{len(self.train_set)}", end="\r")
                x1_batch = x1_batch.to(self.device)  # [batch, 1, 24, 24]
                x2_batch = x2_batch.to(self.device)  # [batch, 1]
                y_batch = y_batch.to(self.device)    # [batch] with class indices 0–2

                self.optimizer.zero_grad()
                outputs = self.model(x1_batch, x2_batch)  # [batch, 3]
                
                loss = self.criterion(outputs, y_batch) 
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x1_batch.size(0)

                 

            avg_loss = running_loss / len(self.train_set.dataset)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Time {time.time() - start:.2f} sec")
    
    def test(self, Model):
    
        Model.eval()
        
        correct = 0
        total = 0
        

        with torch.no_grad():
            start = time.time()
            for x1_batch, x2_batch, y_batch in self.test_set:
        
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = Model(x1_batch, x2_batch)
                _, predicted = torch.max(outputs, 1)
                
                self.all_momenta.append(x2_batch)
                self.all_pred.append(predicted)
                self.all_labels.append(y_batch)
                #[print("Y batch: ", y_batch[i], " argmax " , torch.argmax(y_batch[i],axis=-1),"Pred batch: ",outputs[i], "momenta: " , x2_batch[i]) for i in range(len(predicted))]
                correct += (predicted == torch.argmax(y_batch,axis=-1)).sum().item()
                total += y_batch.size(0)
            
        print(f"Test accuracy: {100 * correct / total:.2f}%, total events: {total:.2f}, Time {time.time() - start:.2f} sec")
        
        self.all_labels = torch.cat(self.all_labels)
        self.all_pred = torch.cat(self.all_pred)
        self.all_momenta = torch.cat(self.all_momenta).squeeze()
        
        self.uniques = np.unique(torch.round(self.all_momenta).cpu().int().numpy())
        
        self.Nentry = np.zeros(shape = (3, len(self.uniques)-1))
        self.acc_e = np.zeros(shape = (3, len(self.uniques)-1))
        
        for particle in range(3):
                for nmom in range(0,len(self.uniques)-1):
                    temp_correct= 0
                    temp_total = 0
                    mask = (torch.argmax(self.all_labels,1) == particle) & (self.all_momenta >= self.uniques[nmom]) & (self.all_momenta < self.uniques[nmom+1])
                    temp_correct += (self.all_pred[mask] == torch.argmax(self.all_labels[mask],axis=-1)).sum().item()
                    temp_total += self.all_labels[mask].size(0)
                    if(temp_total == 0):
                        print(f"Test accuracy by particle: {particle_names[particle]} -- {self.uniques[nmom]} < P < {self.uniques[nmom+1]} GeV/c -- NO EVENTS FOUND" )
                        acc = 0
                    else:
                        acc = temp_correct/temp_total
                        print(f"Test accuracy by particle: {particle_names[particle]} -- {self.uniques[nmom]} < P < {self.uniques[nmom+1]} GeV/c -- acc {100 * temp_correct / temp_total:.2f}% -- total events: {temp_total:.2f}")
                    self.Nentry[particle,nmom] = temp_total
                    self.acc_e[particle, nmom] = acc
                    if(particle == 0):
                        self.labels.append(f"{self.uniques[nmom]} < P < {self.uniques[nmom+1]}")
                
    
    def save_model(self,path):
        self.model.eval()
        example_inputs = (torch.randn(1,1,26,26),torch.randn(1,1))
        example_inputs = tuple(x.to("cpu") for x in example_inputs)
        
        traced_script_module = torch.jit.trace(self.model.to("cpu"),example_inputs)
        
        traced_script_module.save("./model1.pt")
       
    
    def GetPlots(self):
        ByParticleByMomenta(self.acc_e, self.Nentry, self.uniques,self.labels)
        ConfusionPlot(self.all_labels,self.all_pred)
    
  

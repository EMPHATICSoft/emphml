import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

   

def ByParticleByMomenta(acc_e, Nentry, uniques, labels,plot = True):
        
       
        fig, ax = plt.subplots()

        fig.set_size_inches(8, 4) 
        
        im = ax.imshow(acc_e, cmap="coolwarm")


        ax.set_yticks(range(3), labels=["Pion", "Kaon", "Proton"],
                      rotation=45, ha="right", rotation_mode="anchor");

        ax.set_xticks(range(len(uniques)-1), labels=labels,  rotation=45,
                         ha="right", rotation_mode="anchor")

        ax.set_xlim(-0.5,len(labels)-0.5)

        for i in range(3):
            for j in range(len(uniques)-1):
                text = ax.text(j, i, np.round(acc_e[i, j],2),
                               ha="center", va="center", color="k")
        
        plt.savefig('acc.png')
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(8, 4) 
        im1 = ax1.imshow(Nentry, cmap="coolwarm")


        ax1.set_yticks(range(3), labels=["Pion", "Kaon", "Proton"],
                      rotation=45, ha="right", rotation_mode="anchor");

        ax1.set_xticks(range(len(uniques)-1), labels=labels, rotation=45,
                         ha="right", rotation_mode="anchor")
        ax1.set_xlim(-0.5,len(labels)-0.5)

        for i in range(3):
            for j in range(len(uniques)-1):
                text = ax1.text(j, i,int(Nentry[i, j]),
                               ha="center", va="center", color="k")

        plt.savefig('n_entries.png')

def ConfusionPlot(y_true,y_pred):
    
    cm = confusion_matrix(np.argmax(y_true.cpu(),axis=-1), y_pred.cpu(),normalize ='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pion","Kaon","Proton"])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png')

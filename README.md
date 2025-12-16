# emphml
machine learning infrastructure for emphatic 



# What each file does:

*   Model_torch.py: contains model's architecture and forward function
*   Loader_torch.py: loads dataset for training and test
*   Trainer_torch.py: training loop and test loop
*   Plots.py: util to make plots
*   UNet_torch.py: main function, calls the other files/objects

  
# Disclaimer:

This files were made for the arich UNet-CNN, so yeah modify em however needed 


# How to run things:

0. The first sep is to get a torch dataset to train-test on
    * This can be done using ARICHML for emphaticsoft
    * Change the dataset format for your own input shape and needs

1. UNet_torch.py has a parser integrated to run on terminal:
    * run: python UNet_torch.py -h to see all the available options
```
    usage: UNet_torch.py [-h] [-input_files_path INPUT_FILES_PATH] -mod {Test,Train,GetWeights} [-model_path MODEL_PATH]
                     [-n_epochs N_EPOCHS] [-batch BATCH] [-saved_sets SAVED_SETS] [-rebatch REBATCH]

    options:
      -h, --help            show this help message and exit
      -input_files_path INPUT_FILES_PATH, -i INPUT_FILES_PATH
                            Path to input files with TRB3 hits and momenta reconstruction
      -mod {Test,Train,GetWeights}, -m {Test,Train,GetWeights}
                            Mode of the code:  
                             Test: loads a model already trained 
                             Train: makes a model and trains it 
                             GetWeights: get the weights only of a trained model
      -model_path MODEL_PATH, -p MODEL_PATH
                            path to saved model
      -n_epochs N_EPOCHS, -n N_EPOCHS
                            number of epochs
      -batch BATCH, -b BATCH
                            batch size
      -saved_sets SAVED_SETS
                            path of saved sets 
      -rebatch REBATCH      new batch size
```


2. This options allow you to automatically run the training (and/or test) with new/trained weigths
    
3. I'd recomend to test first on jupyter notebook using tensorflow for small/local training and testing and then move to torch and EAF area

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
# Whatever other imports you need

# You can implement classes and helper functions here too.

def sampling(batchsize, df):
    
    tens_labels = [] # list of tuples where each tuple contains a concatenated tensor and its label (0 for not same author, 1 for same author)
    
    for i in range(batchsize):
        #make batches i e if batchsize is 5 => put 5 examples. each example (vector1, vector2, 0/1 (depending on wether or not vecto1 and 2 are by the same author)
        authors = df.Author.unique().tolist() #get unique authors
        first_author = authors.pop(random.randrange(0,len(authors))) # skaffa en random author
        t_f = random.choice([0, 1]) # 0 = not from same author, 1 same author
        if t_f == 1: 
            second_author = first_author
        else: # if t_f is 0 
            second_author = authors.pop(random.randrange(0,len(authors))) # skaffa en annan random author

        author1_tensor = torch.from_numpy(df[df["Author"] == first_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values) # picking a random row where Author = first_author, dropping the Train/Test and Author columns, converting to a ndarray, then to a tensor
        author2_tensor = torch.from_numpy(df[df["Author"] == second_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values)
    
        tensors = torch.cat((author1_tensor, author2_tensor), 0) # concatenating the two author tensors
        
        tens_labels.append((tensors, t_f)) # adding the label of the tensors i e if they're by the same auhtor or not
        
    return tens_labels
        
    
    
class NN(nn.Module):
    def __init__(self, inS, hiS, noL  )
# ins = input size, hs = hidden size, noL = non linearity
        super().__init__()
        if noL != None:
            if noL == 'relu':
                self.noL = nn.ReLU()
            else: #if it's 'tanh'
                self.noL = nn.Tanh()
        if hidden_size != None:      
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, 1)
        else:
            self.l1 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()    
    
    def forward(self, x):
        if self.hiS != None:
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("--batchsize", "-B", dest=B, type=int, default=10, help="The size of the batch passed at each iteration.")
    parser.add_argument("--epochs", "-E", dest=E, type=int, default=3, help="The number of epochs.")
    parser.add_argument("--train_examples", "-TrEx", dest=TrEx, type=int, default=160, help="The number of examples the NN will train on.")
    parser.add_argument("--test_examples", "-TeEx", dest=TeEx, type=int, default=40, help="The number of examples the NN will test on.")
    #parser.add_argument("--hidden_size", "-HS", dest=HS, type=int, default=None, help="The size of the hidden layer.")
    #parser.add_argument("--non_linearity", "-NL", dest=AS, type=str, default=None, choices=['relu', 'tanh'], help="Choice of either ReLU or Tanh".)
    
    # number of iterations is a function of batchsize and train examples: for 160 train examples and batchsize 10: 160/10 = 16 iterations to get through all the training data.  
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    df = pd.read_csv(args.featurefile)
    train_df = df[df["Train/Test"] == "Train"]
    test_df = df[df["Train/Test"] == "Test"]
    test_df.reset_index(inplace=True, drop=True)
    
    print("Done reading.")
    
    inputsize = df.shape[1]-2 # INPUT SIZE TO NN != batchsize
    
    sampling(1, train_df)
    
    # implement everything you need here
    

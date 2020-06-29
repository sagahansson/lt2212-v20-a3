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
    
    for i in range(batchsize):
        #make batches i e if batchsize is 5 => put 5 examples. each example (vector1, vector2, 0/1 (depending on wether or not vecto1 and 2 are by the same author)
        authors = df.Author.unique().tolist() #get unique authors
        first_author = authors.pop(random.randrange(0,len(authors))) # skaffa en random author
        t_f = random.choice([0, 1]) # 0 = not from same author, 1 same author
        if t_f == 1:
            second_author = first_author
        else: # if binary is 0 
            second_author = authors.pop(random.randrange(0,len(authors))) # skaffa en annan random author

        author1_tensor = torch.from_numpy(df[df["Author"].values == first_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values)
        author2_tensor = torch.from_numpy(df[df["Author"].values == second_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values)
    
    print(first_author, second_author)
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    df = pd.read_csv(args.featurefile)
    train_df = df[df["Train/Test"] == "Train"]
    test_df = df[df["Train/Test"] == "Test"]
    test_df.reset_index(inplace=True, drop=True)
    
    print("Done reading.")
    
    inputsize = df.shape[1]-2
    
    
    # implement everything you need here
    

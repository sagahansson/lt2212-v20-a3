import warnings
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, autograd, optim
import random
import torch.nn.functional as F
import sklearn.metrics as metrics
# Whatever other imports you need
warnings.filterwarnings('ignore')


def sampling(batchsize, df):
    
    tens_labels = [] # list of tuples where each tuple contains a concatenated tensor and its label (0 for not same author, 1 for same author)
    
    for i in range(batchsize):
        #make batches i e if batchsize is 5 => put 5 examples. each example (vector1, vector2, 0/1 (depending on wether or not vecto1 and 2 are by the same author)
        authors = df.Author.unique().tolist() #get unique authors
        first_author = authors.pop(random.randrange(0,len(authors))) # get a random author
        t_f = random.choice([0, 1]) # 0 = not from same author, 1 same author
        if t_f == 1: 
            second_author = first_author
        else: # if t_f is 0 
            second_author = authors.pop(random.randrange(0,len(authors))) # skaffa en annan random author

        author1_tensor = torch.FloatTensor(df[df["Author"] == first_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values) # picking a random row where Author = first_author, dropping the Train/Test and Author columns, converting to a ndarray, then to a tensor
        author2_tensor = torch.FloatTensor(df[df["Author"] == second_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values)
        
        tensors = autograd.Variable(torch.FloatTensor((author1_tensor + author2_tensor)))
        
        tens_labels.append((tensors, torch.FloatTensor([t_f]))) # adding the label of the tensors i e if they're by the same auhtor or not
        
    return tens_labels
        
    
    
class NN(nn.Module):
    def __init__(self, inSize, hidSize, nonLin): 
        super().__init__()
        self.hidsize = hidSize
        self.nonlin = nonLin
        if self.hidsize is not None: 
            self.l = nn.Linear(inSize, hidSize)
            self.l2 = nn.Linear(hidSize, 1)
        else:
            self.l = nn.Linear(inSize, 1)    
        if self.nonlin is not None: 
            if self.nonlin == "relu":
                self.nonlin = nn.ReLU()
            else:
                self.nonlin = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l(x)
        if self.nonlin:
            x = self.nonlin(x)
        if self.hidsize:
            x = self.l2(x)
        x = self.sigmoid(x)
        return x
    
def nn_training(traindf, epochs, iterations, batchsize, learning_rate=0.01):
    print("Training...")
    opt = optim.Adam(params=net.parameters(), lr=learning_rate)
    lossfunct = nn.BCELoss()
    for epoch in range(epochs):
        for iteration in range(iterations):
            samples = sampling(batchsize, df)

            tensors = [x[0] for x in samples]
            labels = [x[1] for x in samples]
            tensors = autograd.Variable(torch.stack(tensors))

            labels = autograd.Variable(torch.tensor(labels))
            out = net(tensors)
            
            squeezed = torch.squeeze(out, 1)
            resqueezed = torch.squeeze(squeezed, 1)
            
            loss = lossfunct(resqueezed, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("Training done.")

def nn_testing(testdf, test_sz):
    samples = sampling(test_sz, testdf)
    
    tensors = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    
    tensors = autograd.Variable(torch.stack(tensors))
    preds = []
    
    out = net(tensors)

    squeezed = torch.squeeze(out, 1)
    resqueezed = torch.squeeze(squeezed, 1)

    for value in out:
        if value > 0.5:
            pred = 1
        else:
            pred = 0
        preds.append(pred)

    acc = metrics.accuracy_score(labels, preds)
    prec = metrics.precision_score(labels, preds, average='weighted')
    rec = metrics.recall_score(labels, preds, average='weighted')
    f1 = metrics.f1_score(labels, preds, average='weighted')
    print('Accuracy: ', acc)
    print('Precision: ', prec)
    print('Recall: ', rec)
    print('F1-measure: ', f1)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("--batchsize", "-B", dest="B", type=int, default=10, help="The size of the batch passed at each iteration.")
    parser.add_argument("--epochs", "-E", dest="E", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--train_examples", "-TrEx", dest="TrEx", type=int, default=160, help="The number of random pairs + label the NN will train on.")
    parser.add_argument("--test_examples", "-TeEx", dest="TeEx", type=int, default=40, help="The number of random pairs the NN will test on.")
    parser.add_argument("--hidden_size", "-HS", dest="HS", type=int, default=None, help="The size of the hidden layer.")
    parser.add_argument("--non_linearity", "-NL", dest="NL", type=str, default=None, choices=['relu', 'tanh'], help="Choice of either ReLU or Tanh.")
    

    args = parser.parse_args()
    
    print("Reading {}...".format(args.featurefile))
    df = pd.read_csv(args.featurefile)
    train_df = df[df["Train/Test"] == "Train"]
    test_df = df[df["Train/Test"] == "Test"]
    test_df.reset_index(inplace=True, drop=True)
    print("Done reading.")
    
    
    batchsize = args.B
    epochs = args.E
    TrainEx = args.TrEx
    TestEx = args.TeEx
    iterations = TrainEx//batchsize # number of iterations is a function of batchsize and train examples: for 160 train examples and batchsize 10: 160/10 = 16 iterations to get through all the training data.  
    inputsize = (df.shape[1]-2)
    hidsize = args.HS 
    NonLin = args.NL
    

    net = NN(inputsize, hidsize, NonLin)

    nn_training(train_df, epochs, iterations, batchsize)
    
    nn_testing(test_df, TestEx)

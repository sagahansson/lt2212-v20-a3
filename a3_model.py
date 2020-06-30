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

        author1_tensor = torch.FloatTensor(df[df["Author"] == first_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values) # picking a random row where Author = first_author, dropping the Train/Test and Author columns, converting to a ndarray, then to a tensor
        author2_tensor = torch.FloatTensor(df[df["Author"] == second_author].sample(n=1).drop(["Train/Test", "Author"], axis=1).values)
    
        #tensors = torch.cat((author1_tensor, author2_tensor), 0) # concatenating the two author tensors
        
        tensors = autograd.Variable(torch.FloatTensor((author1_tensor + author2_tensor)))
        
        tens_labels.append((tensors, torch.FloatTensor([t_f]))) # adding the label of the tensors i e if they're by the same auhtor or not
        
    return tens_labels
        
    
    
class NN(nn.Module):
    def __init__(self, inSize=160): # hidSize, nonLin
# inS = input size, hidSize = hidden size, nonLin = non linearity
        super().__init__()
        self.lyr = nn.Linear(inSize, 1)
        self.sigmoid = nn.Sigmoid()
      #  if noL != None:
      #      if noL == 'relu':
      #          self.noL = nn.ReLU()
      #      else: #if it's 'tanh'
      #          self.noL = nn.Tanh()
      #  else:
      #      self.noL = None
      #  if hidden_size != None:      
      #      self.l1 = nn.Linear(input_size, hidden_size)
      #      self.l2 = nn.Linear(hidden_size, 1)
      #  else:
      #      self.l = nn.Linear(input_size, 1)
      #  self.sigmoid = nn.Sigmoid()    
    
    def forward(self, x):
      #  if self.hiS != None:
      #      x = self.l(x)
      #  else:
      #      if self.noL != None:
      #          x = self.l1(x)
      #          x = self.noL(x)
      #          x = self.l2(x)
        x = self.lyr(x)
        x = self.sigmoid(x)
        return x
    
def nn_training(traindf, input_sz=300, epochs=3, iterations=16, batchsize=10, learning_rate=0.01):
    print("Training...")
    net = NN(inSize=input_sz)
    opt = optim.Adam(params=net.parameters(), lr=learning_rate)
    lossfunct = nn.BCELoss()
    for epoch in range(epochs):
        for iteration in range(iterations):
            samples = sampling(batchsize, df)

            tensors = [x[0] for x in samples]
            labels = [x[1] for x in samples]
            tensors = autograd.Variable(torch.stack(tensors))

            labels = autograd.Variable(torch.tensor(labels))
            #print("tensors/target: ", tensors)
            out = net(tensors)
            
            squeezed = torch.squeeze(out, 1)
            resqueezed = torch.squeeze(squeezed, 1)
            
            #print(out)
#            _, prediction = out.max(1)
#            print("prediction: ", prediction)
            #print("target: ", labels)
            loss = lossfunct(resqueezed, labels)
            opt.zero_grad()
            loss.backward()
            #print("loss: ", loss.data)

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
    
    
    
    B = args.B
    E = args.E
    TrEx = args.TrEx
    TeEx = args.TeEx
    iterations = TrEx//B
    inputsize = (df.shape[1]-2) # INPUT SIZE TO NN != batchsize # detta är fel -- inputsize måste vara TrEx dvs the number of random pairs the nn will train on    

    net = NN(inSize=inputsize)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.BCELoss()
        
    
#    for e in range(args.E):
#
#        batch_data = sampling(B, train_df)
#        loss_acc = 0
#        for inputs, label in batch_data:
#            print("inputs: ", inputs)
#            print("label: ", label)
#            optimizer.zero_grad()
#            out = net(inputs)
#            loss = criterion(out, label)
#            loss_acc += loss
#            loss.backward()
#            optimizer.step()
#            print("loss: ", loss.data)
#        if e%50 == 0:
#            print("EPOCH:", e)
#            print(loss_acc.item()/len(batch_data))

    nn_training(train_df, input_sz=inputsize)
    
    nn_testing(test_df, TeEx)

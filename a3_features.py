import os
import sys
import argparse
import numpy as np
import pandas as pd
import glob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
stopwords = stopwords.words('english')

# Whatever other imports you need


def get_files(inputdir):
    # glob to get file, tokenize file, makes each file into a string, appends each string to the list all_files

    enron = glob.glob("{}/*/*.*".format(inputdir)) # enron_sample/bailey-s/1. 
    filenames, authors_in_order = [author.split('/')[-2] + "/" + author.split('/')[-1]  for author in enron], [author.split('/')[-2] for author in enron] # bailey-s/1. & bailey-s
    all_files = []
    for file in enron:
        with open(file, "r") as f:
            onefile = ' '.join(word_tokenize(f.read()))
        all_files.append(onefile)
    return all_files, filenames, authors_in_order

def vectorize_remove(inputdir):
    # vectorize + remove unnecessary words
    all_files, filenames, authors_in_order = get_files(inputdir)

    vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b') # vectorizes + gets removes unalphabetical tokens 
    vectorized = vectorizer.fit_transform(all_files)
    return vectorized, filenames, authors_in_order

def reduce_dims(inputdir, dims):
    # reduce dimensions
    vectorized, filenames, authors_in_order = vectorize_remove(inputdir)

    svd = TruncatedSVD(n_components=dims)
    fit_transformed = svd.fit_transform(vectorized)
    return fit_transformed, filenames, authors_in_order


def create_df(fit_transformed, filenames, authors_in_order, testsize): #inputdir, dims, testsize, outputfile
    # create dataframe, divide randomly to test and train + add column test/train in both, then put test and train df's back together

    df = pd.DataFrame(data=fit_transformed)

    df.insert(0, "Author", authors_in_order)

    test = df.sample(frac=(testsize/100)) 
    train = df.drop(test.index)
    train.insert(0, "Train/Test", (len(train)*["Train"]))
    test.insert(0, "Train/Test", (len(test)*["Test"]))

    df = train.append(test)
    df.reset_index(inplace=True, drop=True)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()
    
    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    
    fit_transformed, filenames, authors_in_order = reduce_dims(args.inputdir, args.dims)
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    create = create_df(fit_transformed, filenames, authors_in_order, args.testsize)
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    write = create.to_csv(args.outputfile, index=False)
    
    print("Done!")
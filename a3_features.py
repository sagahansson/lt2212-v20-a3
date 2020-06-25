import os
import sys
import argparse
import numpy as np
import pandas as pd
import glob
from nltk.tokenize import word_tokenize
import pprint
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.

    print("Done!")
    
    enron = glob.glob("{}/*".format(args.inputdir))
    authors = [os.path.basename(author) for author in enron] # behöver ej?
    texts = [glob.glob("{}/*".format(author)) for author in enron]
    files = [text for authorlist in texts for text in authorlist]
    
    
    
# LT2212 V20 Assignment 3

## Part 1 

A3_features.py is comprised of four functions: get_files, vectorize_remove, reduce_dims and create_df. *Get_files* simply fetches each file, tokenizes it, makes it into a string and appends it to a list. Filenames and author names are also fetched and returned. *Vectorize_remove* vectorizes the files and also removes unnecessary tokens such as stop words and tokens containing anything but alphabetical characters. *Reduce_dims* performs dimensionality reduction on the output of *vectorize_remove*. *Create_df* simply takes the data outputted by *vectorize_remove*, and creates a Pandas Dataframe from it. Part of the dataframe is given the label "test" and the other part "train". Lastly, the dataframe is written to a .csv file.

__How to run:__
The script takes 3 arguments and one optional: inputdir, outputfile, dims and --test. Inputdir represents the root of the directories. Note: the inputdir must be in the same directory as the script is run in. Outputfile decides what the output file will be named. Dims represents the dimensions used in the dimensionality reduction. Test (--test) decides what percentage of the data will consitute the test set. The default value of --test is 20.

Example:
python a3_features.py enron_sample one_file 300


## Part 2

In designing the model, I took inspiraion from various different examples and tutorials found online. The sampling process is done by first fetching a random author from a list of unique authors. A choice between 0 and 1 is then randomised, the result of which will decide whether the feature vectors will come from the same author or not. The vectors are concatenated and, with either 0 or 1, are then appended to a list. The list that is returned represents a batch.

__How to run:__

There is one obligatory argument and six optional arguments available to run the script with. Featurfile is the name of the .csv file produced by a3_features.py. Batchsize (--batchsize, -B) will decide how many vector-label-pairs are passed at each iteration; default is 10. Epochs (--epochs, -E) represents the number of epochs the network will go through; default is 3. Train examples (--train_examples, -TrEx) is the total number of examples the network will train on; default is 160 (mostly for debugging purposes). Test examples (--test_examples, -TeEx) is the same as test examples, but for testing instead; default is 40. 

Example:

python a3_model.py "a_file" -E 10 -TrEx 2320 -TeEx 580

## Part 3

__How to run:__

In part 3, two additional optional arguments are added. Hidden size (--hidden_size, -HS) decides the size of the hidden layer, if using one; default is None, i.e. not using a hidden layer. Non linearity (--non_linearity, -NL) is a choice between ReLU and Tanh. 

Example: python a3_model.py "a_file" -E 5 -TrEx 800 -TeEx 200 -NL relu -HS 25


#import packages and dependencies 
import os
import pandas as pd
from joblib import dump
#from sklearn.externals import joblib
import time
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer


TRAINCOLS = [
    'multiplied list'
]

LABEL = 'Name'


def data_loader(indir, traincols=TRAINCOLS, label=LABEL):
    '''
    One-hot-encodes the pitches, multiplies one-hot column with occurence to provide more nuiance with
    the presence of an note
    Returns a dataframe with features (X) -> the multiplied list of lists, and the labels (y) -> raga name
    '''

    mlb = MultiLabelBinarizer()
    df = pd.read_pickle(indir+'/feature_data.pkl')

    notesOneHot = pd.DataFrame(mlb.fit_transform(df['scale']),columns=mlb.classes_, index=df.index)
    notesOneHot = notesOneHot[['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']]

    df = df.join(notesOneHot)

    #multiply frequency of occurences to each note  
    new_df = df[['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']]
    #create rows from list 
    Row_list =[] 
      
    # Iterate over each row 
    for index, rows in new_df.iterrows(): 
        # Create list for the current row 
        my_list =[rows['A'], rows['A#'],rows['B'],rows['C'],rows['C#'],rows['D'],rows['D#'],
                  rows['E'],rows['F'],rows['F#'],rows['G'],rows['G#']]
          
        # append the list to the final list 
        Row_list.append(my_list) 
    frequencies = list(df['frequencies'])
    df['one hot'] = Row_list

    #multiply the lists
    multiply_lists = []
    for i in range(len(df['frequencies'])): 
        list1 = df['frequencies'][i]
        list2 = df['one hot'][i]
        
        new_frequencies = []
        
        for num1,num2 in zip(list1, list2):
            new_frequencies.append(num1*num2)
        multiply_lists.append(new_frequencies)
    df['multiplied list'] = multiply_lists
    
    X = df[traincols]
    y_value = df[label]
    
    return X,y_value


def train_model(X, y, outdir=None):
    '''
    Inputs features & labels, and trains/outputs model
    '''
    model = GaussianNB()
    model.fit(list(X['multiplied list']), y)

    #creates directory & dumps model 
    if outdir:
        t = int(time.time())
        
        dump(model, os.path.join(outdir, 'naivebayes-model-%d.joblib' % t))

        #joblib.dump(model, os.path.join(outdir, 'naivebayes-model-%d.joblib' % t))

    return model


def driver(indir, outdir=None):
    '''
    Runs all functions
    Inputs dataframe with features and outputs a model to specified directories 
    '''

    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    X, y = data_loader(indir)
    train_model(X, y, outdir)
    return

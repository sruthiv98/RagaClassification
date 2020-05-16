import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os, numpy, nltk, json, scipy, statistics, matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import IPython.display as ipd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from nltk.util import ngrams


def numtonote(num):
    if num == 0:
        return 'C'
    if num == 1:
        return 'C#'
    if num == 2:
        return 'D'
    if num == 3:
        return 'D#'
    if num == 4:
        return 'E'
    if num == 5:
        return 'F'
    if num == 6:
        return 'F#'
    if num == 7:
        return 'G'
    if num == 8:
        return 'G#'
    if num == 9:
        return 'A'
    if num == 10:
        return 'A#'
    if num == 11:
        return 'B'
    
def average_dist(df):
    '''
    Returns the top 7 notes, the frequency of their occurences   
    '''
    y_list = df['y'] #ys for every clip
    sr_list = df['sr']
    hop_length = 512
    
    all_frequencies = []
    all_notes = []
    
    #goes through all clips
    for i in range(len(y_list)): 
        notes = {'C': 0,'C#': 0,'D': 0,'D#': 0,'E': 0,'F': 0,'F#': 0,'G': 0,'G#': 0,'A': 0,'A#': 0,'B': 0}
        
        chromagram = librosa.feature.chroma_stft(y = y_list[i], sr=sr_list[i], hop_length=hop_length)

        notecount = 0 #total number of notes we are getting from clip
        counts = [] #final length should be 12
     
        seq_notes = []
        
        #go through each 1/43rd of a second 
        for row in chromagram.T:
            seq_notes.append(np.argmax(row)) #index of note with highest frequency at that 1/43rd of a second
        clipped_notes = pd.Series(seq_notes).apply(numtonote)
        
        values = []
        for i in range(0, len(clipped_notes),43):
            curr = clipped_notes[i:i+43]
            values.append(mode(curr)[0][0])
        notecount = len(values)
        
        for i in values:   
            if i in notes:
                notes[i]+=1
            else: 
                notes[i] = 1
        
        for note in notes:
            notes[note] = notes[note]/notecount
        
        notedf = pd.DataFrame.from_dict(notes, orient = 'index')
        frequencies, notes = top_7_notes(notedf)
        all_frequencies.append(frequencies)
        all_notes.append(notes)

    return [all_frequencies, all_notes]

def top_7_notes(df):
    frequencies = list(df.sort_index()[0])
    notes = list(df.sort_values(by=0,ascending=False).rename(columns={0:'Notes'})[:7].index)
    return [frequencies, notes]


def make_features(indir=None,outdir=None):
    df = pd.read_pickle(indir+'/cleaned_data.pkl')
    freq_and_notes = average_dist(df)
    df['frequencies'] = freq_and_notes[0]
    df['scale'] = freq_and_notes[1]

    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
   
    df.to_pickle(os.path.join(outdir,'feature_data.pkl'))
    
    return  

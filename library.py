#library functions

#import libraries and packages
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os, nltk, json, scipy, statistics
import pandas as pd
from scipy.stats import mode
import IPython.display as ipd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from nltk.util import ngrams

#EDA

def average_dist(df):
    y_list = df['y'] #ys for every clip
    sr_list = df['sr']
    hop_length = 512
    notes = {'C': 0,'C#': 0,'D': 0,'D#': 0,'E': 0,'F': 0,'F#': 0,'G': 0,'G#': 0,'A': 0,'A#': 0,'B': 0}
    global_notecount = 0 
    
    #chromagram color code 
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([20/256, 24/256, 64/256, 1])
    newcolors[:150, :] = pink
    newcmp = ListedColormap(newcolors)
    
    chromo_y_list = []
    chromo_sr_list = []

    #goes through all clips
    for i in range(len(y_list)): 
        if i < 79: 
            chromo_y_list.append(y_list[i])
            chromo_sr_list.append(sr_list[i])
        chromo_y_list_flat = [item for sublist in chromo_y_list for item in sublist]
        
        chromagram = librosa.feature.chroma_stft(y = y_list[i], sr=sr_list[i], hop_length=hop_length)

        if i == 80: #80000 is 40 second
            chromagram_new = librosa.feature.chroma_stft(y = np.asarray(chromo_y_list_flat), sr=sr_list[0], hop_length=hop_length)
            plt.figure(figsize=(21, 5))
            librosa.display.specshow(chromagram_new, x_axis='time', y_axis='chroma', hop_length=512, cmap=newcmp)
            plt.colorbar(format='%+2.0f dB')
            plt.title(df['Name'][0])
        
        notecount = 0 #total number of notes we are getting from clip
        counts = [] #final length should be 12
     
        seq_notes = []
        #go through each 1/43rd of a second 
       # print(len(chromagram.T),'chromagram.T should be large number')
        for row in chromagram.T:
            seq_notes.append(np.argmax(row)) #index of note with highest frequency at that 1/43rd of a second
        all_notes = pd.Series(seq_notes).apply(numtonote)
    
        values = []
        for i in range(0, len(all_notes),86):
            curr = all_notes[i:i+86]
            values.append(mode(curr)[0][0])
        notecount = len(values)
        global_notecount += notecount
        
        #add notecount to global dictionary directly
        for i in values:   
            if i in notes:
                notes[i]+=1
            else: 
                notes[i] = 1

    for note in notes:
        notes[note] = notes[note]/global_notecount
        
    notedf = pd.DataFrame.from_dict(notes, orient = 'index')
    return notedf


def EDA(cleaned_df):
    '''
    Input: cleaned dataframe (run clean data target)
    Output: plots for EDA

    Performs analysis of frequency of notes, bigrams, and trigrams.
    Generates plots for EDA of cleaned data. 
    ''' 

    ngrams_dfs = create_ngrams(cleaned_df)
    histogram_df = average_dist(cleaned_df)
    eda = histogram_df.sort_values(by=0,ascending=False).rename(columns={0:'Notes'})
    plot = eda.sort_values(by='Notes',ascending=False)[:7].plot(kind = 'bar',
                                                                title = 'Note Frequency Counts for '+ cleaned_df['Name'][0][:-5])

    plot.set_xlabel("Notes")
    plot.set_ylabel("Frequency")
    
    return [plot, ngrams_dfs]




#bigram/trigram/ngram analysis


def numtonote(num):
    '''
    Turns number into corresponding note on scale. To be used with array positions on chromogram. 
    '''
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

#bigram/trigram/ngram analysis

def create_ngrams(cleaned_df):
    '''
    Input: cleaned dataframe
    Output: list of lists for bigram and trigram plots
    '''
    y_list = []
    sr_list = []
    for i in range(0,len(cleaned_df['y']),100):
        try:
            small_list = [item for sublist in list(cleaned_df['y'][i:i+100]) for item in sublist]
            y_list.append(np.asarray(small_list))
            sr_list.append(cleaned_df['sr'][i])
        except: 
            continue
    
    #holds counts for bigrams and trigrams for whole song
    ragabigrams = {}
    ragatrigrams = {}
    
    for i in range(len(y_list)):
        chromagram = librosa.feature.chroma_stft(y = y_list[i], sr=sr_list[i], hop_length=512)
        #helper function return

        #maps bigram with its counts
        bigramdict = extract_ngrams(chromagram,2)
       
        for key in bigramdict:
            if key in ragabigrams:
                ragabigrams[key] += bigramdict[key]
                
            else:
                ragabigrams[key] = bigramdict[key]
        
        #maps trigram with its counts
        trigramdict = extract_ngrams(chromagram,3)
        for key in trigramdict:
            if key in ragatrigrams:
                ragatrigrams[key] += trigramdict[key]
                
            else:
                ragatrigrams[key] = trigramdict[key]
    
    #plots bigrams
    bdf = pd.DataFrame({'Bigrams': list(ragabigrams.keys()), 'Counts': list(ragabigrams.values())})
    plot = bdf.sort_values(by = 'Counts', ascending = False)[:10].plot(kind = 'bar', x= 'Bigrams', y = 'Counts', title = cleaned_df['Name'][0]) 
    plot.set_xlabel("Bigrams")
    plot.set_ylabel("Counts")
    
    #plots trigrams
    tdf = pd.DataFrame({'Trigrams': list(ragatrigrams.keys()), 'Counts': list(ragatrigrams.values())})
    plot2 = tdf.sort_values(by = 'Counts', ascending = False)[:10].plot(kind = 'bar', x= 'Trigrams', y = 'Counts', title = cleaned_df['Name'][0]) 
    plot2.set_xlabel("Trigrams")
    plot2.set_ylabel("Counts")

    return [bdf,tdf]


def extract_ngrams(chromagram, num):
    '''
    Input: chromagram and ngram number (2 for bigram, 3 for trigram, etc)
    Output: dictionary mapping of bigrams and trigrams
    '''

    #get loudest notes in chromagram
    seq_notes = []
    max_val = []
    for row in chromagram.T:
        seq_notes.append(np.argmax(row))
        max_val.append(np.max(row))
    all_notes = pd.Series(seq_notes).apply(numtonote)

    #get notes for every half of a second
    values = []
    for i in range(0, len(all_notes),22):
        curr = all_notes[i:i+22]
        values.append(mode(curr)[0][0])

    #removing consecutive notes (not taking repeats of notes that are held for a long time)
    newvals = []
    for i in range(len(values)):
        if i != 0 and values[i] == values[i-1]:
            continue
        else:
            newvals.append(values[i])
        
    #generate ngrams from notes
    n_grams = ngrams(newvals, num)
   
    fdist = nltk.FreqDist(n_grams)
    ngramlist = []
    counts = []
    for k,v in fdist.items():
        ngramlist.append(k)
        counts.append(v)

    #dictionary of ngrams and number of occurences
    clipdict = {ngramlist[i]: counts[i] for i in range(len(ngramlist))}
    return clipdict


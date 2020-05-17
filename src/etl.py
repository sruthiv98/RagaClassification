#import packages and dependencies
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os, numpy, scipy, statistics, matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import json


def load(indir=None,outdir=None):
    '''
    Loads data from indir with Librosa
    Organizes it into a dataframe
    Outputs dataframe to a pkl file
    '''
    songs = os.listdir(indir)
    y_array = []
    sr_array = []
    song_name = []
    for i in songs:
        if '.DS_Store' in i: 
            continue
        y, sr = librosa.load(indir+'/'+ i) #load song 
        y_array.append(y)
        sr_array.append(sr)
        song_name.append(i)
    df = pd.DataFrame({'Name': song_name,
                       'y': y_array,
                       'sr': sr_array})
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
        
    df.to_pickle(os.path.join(outdir,r'loaded_data.pkl'))

    return None



def clean_data(indir = None, outdir = None):
    '''
    Takes the loaded data, converts pitch into C, and
    chunks the clips into subclips
    Outputs a pkl file with dataframe of cleaned and chunked data 
    '''

    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df = pd.read_pickle(indir+'/loaded_data.pkl')

    dfdict = {'Name': [], 'y': [], 'sr': []}
    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    
    audio = list(df['Name'])
    ys = np.asarray(list(df['y']))
    srs = list(df['sr'])
    
    pitchdict = {'Asavari Natabhairavi 1': 'C#', 'Asavari Natabhairavi 2': 'D',
                 'Asavari Natabhairavi 3': 'D',
               'Bhairav Mayamalavagowlai 1': 'C#', 'Bhairav Mayamalavagowlai 2': 'C', 
               'Bhairav Mayamalavagowlai 3': 'C#', 'Bhairavi Hanumatodi 1': 'D',
               'Bhairavi Hanumatodi 2': 'C', 'Bhairavi Hanumatodi 3': 'D#',
               'Bilawal Dheerashankarabharanam 1': 'D', 'Bilawal Dheerashankarabharanam 2': 'D#',
               'Bilawal Dheerashankarabharanam 3': 'E', 'Kafi Karaharapriya 1': 'D',
               'Kafi Karaharapriya 2': 'D', 'Kafi Karaharapriya 3': 'D#',
               "Kalyan Kalyani 1": 'E', "Kalyan Kalyani 2": 'D#', 'Kalyan Kalyani 3': 'D#',
               'Khamaj Harikambhoji 1': 'D', 'Khamaj Harikambhoji 2': 'E', 'Khamaj Harikambhoji 3': 'E',
               'Marva Gamanasharma 1': 'C', 'Marva Gamanasharma 2': 'E', 'Marva Gamanasharma 3': 'D',
               'Poorvi Kamavardhani 1': 'C#', 'Poorvi Kamavardhani 2': 'C#', 'Poorvi Kamavardhani 3': 'C#',
               'Todi Subhapantuvarali 1': 'C', 'Todi Subhapantuvarali 2': 'C#', 'Todi Subhapantuvarali 3': 'D'}
    
    #iterate through input dict
    for i in range(len(df)):
        y = ys[i]
        sr = srs[i]
        
        #check pitch of each input, convert it to C if it's not      
        pitch = pitchdict[audio[i][:-4]]
        
        if pitch == 'C':
            newy = y
        elif pitch == 'C#':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-1)
        elif pitch == 'D':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-2)
        elif pitch == 'D#':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-3)
        elif pitch == 'E':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-4)
        elif pitch == 'F':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-5)
        elif pitch == 'F#':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-6)
        elif pitch == 'G':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=-7)
        elif pitch == 'G#':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=+4)
        elif pitch == 'A':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=+3)
        elif pitch == 'A#':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=+2)
        elif pitch == 'B':
            newy = librosa.effects.pitch_shift(y, sr, n_steps=+1)
        
        y = newy
        ylist = []

        #chunk the data to create smaller subclips
        first = 0
        second = 100000
        while second < len(y):
            ylist.append(y[first:second])
            first = second
            second = second + 100000

        cliptitles = []
        
        for j in range(len(ylist)):
            string = audio[i][:-6] #clean string to show raga only 
            cliptitles.append(string)
            
        for k in range(len(cliptitles)):
            dfdict['Name'].append(cliptitles[k])
            dfdict['y'].append(ylist[k])

            dfdict['sr'].append(sr)
            
    cleaned = pd.DataFrame(dfdict)

    #create output directory
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    #output cleaned file
    cleaned.to_pickle(os.path.join(outdir,'cleaned_data.pkl'))
    
    return cleaned


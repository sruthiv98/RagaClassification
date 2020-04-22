import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os, numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
from IPython.display import Audio
import pandas as pd



def load(indir=None,outdir=None): 
    songs = os.listdir(indir)
    y_array = []
    sr_array = []
    song_name = []
    for i in songs:
        if '.DS_Store' in i: 
            continue
        y, sr = librosa.load(indir+'/'+ i)
        y_array.append(y)
        sr_array.append(sr)
        song_name.append(i)
    df = pd.DataFrame({'Name': song_name,
                       'y': y_array,
                       'sr': sr_array})
    df.to_pickle(os.path.join(outdir,r'loaded_data.pkl'))

    return None


def compute_pitch(chromagram):
    total_occurences_above_8 = []
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    for i in chromagram: 
        count = 0
        for j in i: 
            if j >=0.7: 
                count+=1
        total_occurences_above_8.append(count)
        
    df = pd.DataFrame({'occurences above 0.8':total_occurences_above_8,'notes':notes})
    df = df.sort_values('occurences above 0.8',ascending=False)
    pitch = df.iloc[0]['notes']
        
    return pitch

def get_clean_data(df):
    dfdict = {'songs': [], 'y': [], 'sr': []}
    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    
    audio = list(df['Name'])
    ys = np.asarray(list(df['y']))
    srs = list(df['sr'])
    
    #iterate through input dict
    for i in range(len(df)):
        y = ys[i]
        sr = srs[i]
        
        #check pitch of each input
        chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hop_length)
        
        pitch = compute_pitch(chromagram)
        
        if pitch == 'C':
            continue
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
        first = 0
        second = 500000
        while second < len(y):
            ylist.append(y[first:second])
            first = second
            second = second + 500000
            
        cliptitles = []
        
        for j in range(len(ylist)):
            string = audio[i][:-6]
            cliptitles.append(string)
            
        for k in range(len(cliptitles)):
            dfdict['songs'].append(cliptitles[k])
            dfdict['y'].append(ylist[k])
            #assuming sr does not change?
            dfdict['sr'].append(sr)
            
    newdf = pd.DataFrame(dfdict)
    
    return newdf



def clean_data(indir = None, outdir = None):

    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df = pd.read_pickle('data/loaded_data.pkl')
    
    
    dfdict = {'songs': [], 'y': [], 'sr': []}
    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    
    audio = list(df['Name'])
    ys = df['y']
    newys = [i for i in ys]
    ys = newys
    srs = list(df['sr'])
    
    #iterate through input dict
    for i in range(len(df)):
        y = ys[i]
        sr = srs[i]
        
        #check pitch of each input
        chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hop_length)
        
        pitch = compute_pitch(chromagram)
        
        if pitch == 'C':
            continue
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
        first = 0
        second = 50000
        print(len(y))
        while second < len(y):
            ylist.append(y[first:second])
            first = second
            second = second + 50000
        cliptitles = []
        
        for j in range(len(ylist)):
            string = audio[i][:-6]
            cliptitles.append(string)
            
        for k in range(len(cliptitles)):
            dfdict['songs'].append(cliptitles[k])
            dfdict['y'].append(ylist[k])
            #assuming sr does not change?
            dfdict['sr'].append(sr)
            
    cleaned = pd.DataFrame(dfdict)
    
    #cleaned = get_clean_data(df)
   
    cleaned.to_pickle(os.path.join(outdir,r'cleaned_data.pkl'))
    
    return 


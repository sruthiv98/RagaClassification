#library functions

#EDA

def frequency_counts(df):
    y_list = df['y'] #ys for every clip
    sr_list = df['sr']
    hop_length = 512
    notes = {'C': 0,'C#': 0,'D': 0,'D#': 0,'E': 0,'F': 0,'F#': 0,'G': 0,'G#': 0,'A': 0,'A#': 0,'B': 0}
    global_notecount = 0 
    #goes through all clips
    for i in range(len(y_list)): 
        
        chromagram = librosa.feature.chroma_stft(y = y_list[i], sr=sr_list[i], hop_length=hop_length)
        
        notecount = 0 #total number of notes we are getting from clip
        counts = [] #final length should be 12
     
        seq_notes = []
        #go through each 1/43rd of a second 
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
        #takes average frequency for each note
        notes[note] = notes[note]/global_notecount
        
    notedf = pd.DataFrame.from_dict(notes, orient = 'index')
    return notedf

def frequency_plot_generator(songs): 
    #input should be list of x songs
    #songs = [song1, song2, song3]
    y_array = []
    sr_array = []
    song_name = []
    for i in songs:
        if '.DS_Store' in i: 
            continue
        y,sr = librosa.load('test_data_raw/'+ i)
        y_array.append(y)
        sr_array.append(sr)
        song_name.append(i)

    df = pd.DataFrame({'Name': song_name,
                       'y': y_array,
                       'sr': sr_array})
    
    cleaned_df = get_clean_data(df)
    
    histogram_df = frequency_counts(cleaned_df)

    #generate plots
    plot = histogram_df.sort_values(by=[0])[:7].sort_index().rename(columns={0:'Notes'}).plot(kind = 'bar', title = songs[0][:-6])
    plot.set_xlabel("Notes")
    plot.set_ylabel("Frequency")
    
    return [plot,histogram_df]


#bigram/trigram/ngram analysis


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


def create_ngrams(songs):
    y_array = []
    sr_array = []
    song_name = []
    for i in songs:
        if '.DS_Store' in i: 
            continue
        y,sr = librosa.load('test_data_raw/'+ i)
        y_array.append(y)
        sr_array.append(sr)
        song_name.append(i)

    df = pd.DataFrame({'Name': song_name,
                       'y': y_array,
                       'sr': sr_array})
    
    cleaned_df = get_clean_data(df)
    #gives us df of clip title (raga name), y, sr
    
    y_list = cleaned_df['y']
    sr_list = cleaned_df['sr']
    
    ragabigrams = {}
    ragatrigrams = {}
    
    for i in range(len(y_list)):
        chromagram = librosa.feature.chroma_stft(y = y_list[i], sr=sr_list[i], hop_length=512)
        if i == 2: 
            plt.figure(figsize=(17, 5))
            librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
            
        
        #helper function return
        bigramdict = extract_ngrams(chromagram,2)
        for key in bigramdict:
            if key in ragabigrams:
                ragabigrams[key] += bigramdict[key]
                
            else:
                ragabigrams[key] = bigramdict[key]
                
                bigramdict = extract_ngrams(chromagram,2)
        
        trigramdict = extract_ngrams(chromagram,3)
        for key in trigramdict:
            if key in ragatrigrams:
                ragatrigrams[key] += trigramdict[key]
                
            else:
                ragatrigrams[key] = trigramdict[key]
        
    
    bdf = pd.DataFrame({'Bigrams': list(ragabigrams.keys()), 'Counts': list(ragabigrams.values())})
    plot = bdf.sort_values(by = 'Counts', ascending = False)[:10].plot(kind = 'bar', x= 'Bigrams', y = 'Counts', title = songs[0][:-6]) 
    plot.set_xlabel("Bigrams")
    plot.set_ylabel("Counts")
    
    tdf = pd.DataFrame({'Trigrams': list(ragatrigrams.keys()), 'Counts': list(ragatrigrams.values())})
    plot2 = tdf.sort_values(by = 'Counts', ascending = False)[:10].plot(kind = 'bar', x= 'Trigrams', y = 'Counts', title = songs[0][:-6]) 
    plot2.set_xlabel("Trigrams")
    plot2.set_ylabel("Counts")
    
    return None


def extract_ngrams(chromagram, num):
    seq_notes = []
    max_val = []
    for row in chromagram.T:
        seq_notes.append(np.argmax(row))
        max_val.append(np.max(row))
    all_notes = pd.Series(seq_notes).apply(numtonote)

    values = []
    for i in range(0, len(all_notes),22):
        curr = all_notes[i:i+22]
        values.append(mode(curr)[0][0])

    newvals = []
    for i in range(len(values)):
        if i != 0 and values[i] == values[i-1]:
            continue
        else:
            newvals.append(values[i])
        
    n_grams = ngrams(newvals, num)
    fdist = nltk.FreqDist(n_grams)
    ngramlist = []
    counts = []
    for k,v in fdist.items():
        ngramlist.append(k)
        counts.append(v)
    clipdict = {ngramlist[i]: counts[i] for i in range(len(ngramlist))}

    return clipdict

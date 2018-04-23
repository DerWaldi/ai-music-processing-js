import glob, os, argparse

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# parse chord names to indices
def strCtnsPatt(text, pattern):
    return pattern in text

def parse_chord_label(text):
    if strCtnsPatt(text, 'N'):
        chordId = 1    
    elif strCtnsPatt(text, 'C:min'):
        chordId = 14
    elif strCtnsPatt(text, 'C#:min') or strCtnsPatt(text, 'Db:min'):
        chordId = 15
    elif strCtnsPatt(text, 'D:min'):
        chordId = 16
    elif strCtnsPatt(text, 'D#:min') or strCtnsPatt(text, 'Eb:min'):
        chordId = 17
    elif strCtnsPatt(text, 'E:min'):
        chordId = 18
    elif strCtnsPatt(text, 'F:min'):
        chordId = 19
    elif strCtnsPatt(text, 'F#:min') or strCtnsPatt(text, 'Gb:min'):
        chordId = 20
    elif strCtnsPatt(text, 'G:min'):
        chordId = 21
    elif strCtnsPatt(text, 'G#:min') or strCtnsPatt(text, 'Ab:min'):
        chordId = 22
    elif strCtnsPatt(text, 'A:min'):
        chordId = 23
    elif strCtnsPatt(text, 'A#:min') or strCtnsPatt(text, 'Bb:min'):
        chordId = 24
    elif strCtnsPatt(text, 'B:min'):
        chordId = 25   
    elif strCtnsPatt(text, 'C:maj'):
        chordId = 2  
    elif strCtnsPatt(text, 'C#:maj') or strCtnsPatt(text, 'Db:maj'):
        chordId = 3  
    elif strCtnsPatt(text, 'D:maj'):
        chordId = 4
    elif strCtnsPatt(text, 'D#:maj') or strCtnsPatt(text, 'Eb:maj'):
        chordId = 5
    elif strCtnsPatt(text, 'E:maj'):
        chordId = 6
    elif strCtnsPatt(text, 'F:maj'):
        chordId = 7
    elif strCtnsPatt(text, 'F#:maj') or strCtnsPatt(text, 'Gb:maj'):
        chordId = 8
    elif strCtnsPatt(text, 'G:maj'):
        chordId = 9
    elif strCtnsPatt(text, 'G#:maj') or strCtnsPatt(text, 'Ab:maj'):
        chordId = 10
    elif strCtnsPatt(text, 'A:maj'):
        chordId = 11
    elif strCtnsPatt(text, 'A#:maj') or strCtnsPatt(text, 'Bb:maj'):
        chordId = 12
    elif strCtnsPatt(text, 'B:maj'):
        chordId = 13
    elif strCtnsPatt(text, 'Cmin'):
        chordId = 14  
    elif strCtnsPatt(text, 'C#min') or strCtnsPatt(text, 'Dbmin'):
        chordId = 15  
    elif strCtnsPatt(text, 'Dmin'):
        chordId = 16
    elif strCtnsPatt(text, 'D#min') or strCtnsPatt(text, 'Ebmin'):
        chordId = 17
    elif strCtnsPatt(text, 'Emin'):
        chordId = 18
    elif strCtnsPatt(text, 'Fmin'):
        chordId = 19
    elif strCtnsPatt(text, 'F#min') or strCtnsPatt(text, 'Gbmin'):
        chordId = 20
    elif strCtnsPatt(text, 'Gmin'):
        chordId = 21
    elif strCtnsPatt(text, 'G#min') or strCtnsPatt(text, 'Abmin'):
        chordId = 22
    elif strCtnsPatt(text, 'Amin'):
        chordId = 23
    elif strCtnsPatt(text, 'A#min') or strCtnsPatt(text, 'Bbmin'):
        chordId = 24
    elif strCtnsPatt(text, 'Bmin'):
        chordId = 25
    elif strCtnsPatt(text, 'Cmaj'):
        chordId = 2  
    elif strCtnsPatt(text, 'C#maj') or strCtnsPatt(text, 'Dbmaj'):
        chordId = 3  
    elif strCtnsPatt(text, 'Dmaj'):
        chordId = 4
    elif strCtnsPatt(text, 'D#maj') or strCtnsPatt(text, 'Ebmaj'):
        chordId = 5
    elif strCtnsPatt(text, 'Emaj'):
        chordId = 6
    elif strCtnsPatt(text, 'Fmaj'):
        chordId = 7
    elif strCtnsPatt(text, 'F#maj') or strCtnsPatt(text, 'Gbmaj'):
        chordId = 8
    elif strCtnsPatt(text, 'Gmaj'):
        chordId = 9
    elif strCtnsPatt(text, 'G#maj') or strCtnsPatt(text, 'Abmaj'):
        chordId = 10
    elif strCtnsPatt(text, 'Amaj7'):
        chordId = 11
    elif strCtnsPatt(text, 'A#maj') or strCtnsPatt(text, 'Bbmaj'):
        chordId = 12
    elif strCtnsPatt(text, 'Bmaj'):
        chordId = 13   
    elif strCtnsPatt(text, 'Cm'):
        chordId = 14  
    elif strCtnsPatt(text, 'C#m') or strCtnsPatt(text, 'Dbm'):
        chordId = 15  
    elif strCtnsPatt(text, 'Dm'):
        chordId = 16
    elif strCtnsPatt(text, 'D#m') or strCtnsPatt(text, 'Ebm'):
        chordId = 17
    elif strCtnsPatt(text, 'Em'):
        chordId = 18
    elif strCtnsPatt(text, 'Fm'):
        chordId = 19
    elif strCtnsPatt(text, 'F#m') or strCtnsPatt(text, 'Gbm'):
        chordId = 20
    elif strCtnsPatt(text, 'Gm'):
        chordId = 21
    elif strCtnsPatt(text, 'G#m') or strCtnsPatt(text, 'Abm'):
        chordId = 22
    elif strCtnsPatt(text, 'Am'):
        chordId = 23
    elif strCtnsPatt(text, 'A#m') or strCtnsPatt(text, 'Bbm'):
        chordId = 24
    elif strCtnsPatt(text, 'Bm'):
        chordId = 25
    elif strCtnsPatt(text, 'C#') or strCtnsPatt(text, 'Db'):
        chordId = 3  
    elif strCtnsPatt(text, 'C'):
        chordId = 2  
    elif strCtnsPatt(text, 'D#') or strCtnsPatt(text, 'Eb'):
        chordId = 5
    elif strCtnsPatt(text, 'D'):
        chordId = 4
    elif strCtnsPatt(text, 'E'):
        chordId = 6
    elif strCtnsPatt(text, 'F#') or strCtnsPatt(text, 'Gb'):
        chordId = 8
    elif strCtnsPatt(text, 'F'):
        chordId = 7
    elif strCtnsPatt(text, 'G#') or strCtnsPatt(text, 'Ab'):
        chordId = 10
    elif strCtnsPatt(text, 'G'):
        chordId = 9
    elif strCtnsPatt(text, 'A#') or strCtnsPatt(text, 'Bb'):
        chordId = 12
    elif strCtnsPatt(text, 'A'):
        chordId = 11
    elif strCtnsPatt(text, 'B'):
        chordId = 13
    return chordId - 1

# extract features
def extract_features(filename, frame_size=2048, hop_length=512):
    filename_mp3 = os.path.splitext(filename)[0] + ".mp3"
    
    # generate chromagram
    y, sr = librosa.load(filename_mp3)
    S = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=frame_size, hop_length=hop_length)
    feature_step = hop_length / sr
    chroma_length = chroma.shape[1]

    # load chord labels
    df = pd.read_csv(filename, delimiter='\t', header=None)
    
    # zip both together 
    x_total = np.zeros((chroma_length, chroma.shape[0]))
    y_total = np.zeros((chroma_length, 1))

    for i in range(chroma_length):
        time_sec = i * feature_step
        label = 0
        rows_before = df[df[0] <= time_sec]
        if len(rows_before) > 0:
            label = parse_chord_label(rows_before.iloc[-1][2])        
        x_total[i, :] = chroma[:, i]
        y_total[i] = label
        
    return x_total, y_total

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser(description='Extracts chords from beatles songs.')
    parser.add_argument("--src_files", default='/media/sebi/Annotations/chords/beatles*.chords', type=str, help="Search pattern for chord annotation files")
    parser.add_argument("--hop_length", default=512, type=int, help="hop size of chromagram")
    parser.add_argument("--frame_size", default=2048, type=int, help="fft size of chromagram")
    parser.add_argument("--dataset_filename", default='dataset.npz', type=str, help="filename the features should be stored at (*.npz)")
    args = parser.parse_args()
    
    # list all files in dataset directory
    files = glob.glob(args.src_files)

    # split into train and test subsets
    train_files, test_files = train_test_split(files, test_size=0.10, random_state=42)
    print("train_files: {} \t test_files: {}".format(len(train_files), len(test_files)))

    x_train = None
    y_train = None

    print("generating train split")
    for filename in tqdm(train_files):
        x_song, y_song = extract_features(filename, frame_size=args.frame_size, hop_length=args.hop_length)
        # append to training set
        if x_train is None:
            x_train = x_song
            y_train = y_song
        else:
            x_train = np.concatenate((x_train, x_song))
            y_train = np.concatenate((y_train, y_song))


    x_test = None
    y_test = None

    print("generating test split")
    for filename in tqdm(test_files):
        x_song, y_song = extract_features(filename, frame_size=args.frame_size, hop_length=args.hop_length)
        # append to training set
        if x_test is None:
            x_test = x_song
            y_test = y_song
        else:
            x_test = np.concatenate((x_test, x_song))
            y_test = np.concatenate((y_test, y_song))
            
    # store dataset
    np.savez_compressed(args.dataset_filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
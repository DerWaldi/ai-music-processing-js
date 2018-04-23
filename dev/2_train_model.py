import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical  

# hyperparameters
max_seq_size = 32
batch_size = 128
epoch_count = 1

dataset = np.load('dataset_min.npz')
X_train_orig_raw, Y_train_orig_raw, X_test_orig, Y_test_orig = dataset['x_train'], dataset['y_train'], \
    dataset['x_test'], dataset['y_test']
    
Y_train_orig_raw = to_categorical(Y_train_orig_raw, num_classes=25)
Y_test_orig = to_categorical(Y_test_orig, num_classes=25)

for d in [X_train_orig_raw, X_test_orig, Y_train_orig_raw, Y_test_orig]:
    print(d.shape)

# shift circularly to balance chords
X_train_orig = X_train_orig_raw
Y_train_orig = Y_train_orig_raw
for i in range(1,12):
    X_rolled = np.roll(X_train_orig_raw, i)
    Y_rolled = np.concatenate((Y_train_orig_raw[:,0:1], np.roll(Y_train_orig_raw[:, 1:13], i), np.roll(Y_train_orig_raw[:, 13:25], i)), axis=1)
    X_train_orig = np.concatenate((X_train_orig, X_rolled), axis=0)
    Y_train_orig = np.concatenate((Y_train_orig, Y_rolled), axis=0)
    
feature_count = X_train_orig.shape[1]
target_count = Y_train_orig.shape[1]  

def normalize(X):
    return (X.astype('float32') - 120) / (X.shape[1] - 120)

#X_train_orig = normalize(X_train_orig)
#X_test_orig = normalize(X_test_orig)
    
for d in [X_train_orig, X_test_orig, Y_train_orig, Y_test_orig]:
    print(d.shape)
    
# we discard some frames (better would be to pad)

def cut_sequences(a, max_seq_size):
    n = len(a)
    n_cut = len(a) - len(a) % max_seq_size
    return a[:n_cut].reshape(-1, max_seq_size, a.shape[1])

X_train_seq = cut_sequences(X_train_orig, max_seq_size)
X_test_seq = cut_sequences(X_test_orig, max_seq_size)
Y_train_seq = cut_sequences(Y_train_orig, max_seq_size)
Y_test_seq = cut_sequences(Y_test_orig, max_seq_size)

for d in [X_train_seq, X_test_seq, Y_train_seq, Y_test_seq]:
    print(d.shape)
    
Y_train_flat = Y_train_seq.reshape(-1, target_count)
Y_test_flat = Y_test_seq.reshape(-1, target_count)

X_train_seq_conv = X_train_seq.reshape(X_train_seq.shape[0], max_seq_size, feature_count, 1)
X_test_seq_conv = X_test_seq.reshape(X_test_seq.shape[0], max_seq_size, feature_count, 1)

X_train = X_train_seq_conv
X_valid = X_test_seq_conv
Y_train = Y_train_seq
Y_valid = Y_test_seq

model = Sequential()

model.add(TimeDistributed(Dense(12, activation='relu'), input_shape=(max_seq_size, feature_count, 1)))
model.add(TimeDistributed(Flatten()))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(25, activation='softmax')))

#model.add(LSTM(50, batch_input_shape=(batch_size, max_seq_size, feature_count), return_sequences=True))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(LSTM(100, return_sequences=True))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

# no conv1D for kerasJS (https://github.com/transcranial/keras-js/issues/79)
#model.add(TimeDistributed(Convolution1D(32, 3, activation='relu'), input_shape=(max_seq_size, feature_count, 1)))
#model.add(TimeDistributed(Convolution1D(32, 3, activation='relu')))
#model.add(TimeDistributed(MaxPooling1D(2, 2)))
#model.add(Dropout(0.25))
#model.add(TimeDistributed(Flatten()))
#model.add(BatchNormalization())
#model.add(Bidirectional(LSTM(128, batch_input_shape=(batch_size, feature_count, 1))))
#model.add(Bidirectional(LSTM(128, return_sequences=True)))
#model.add(Dropout(0.25))
#model.add(TimeDistributed(Dense(25, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('param count:', model.count_params())
print('input shape:', model.input_shape)
print('output shape:', model.output_shape)

tbCallBack = TensorBoard(log_dir='./logs')
mcCallBack = ModelCheckpoint('model-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

hist = model.fit(X_train, Y_train,
          validation_data=(X_valid, Y_valid),
          epochs=epoch_count, batch_size=batch_size,
          callbacks=[tbCallBack])

from keras.models import load_model
model.save('chord_model.h5')
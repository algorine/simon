from Simon.DataGenerator import *
from Simon.Encoder import *
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge, Masking
from keras.layers import Embedding, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os
import time
import pickle

class Simon:
    def __init__(self,encoder):
        self.encoder = encoder

    def binarize(self,x, sz=71):
        return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))
    
    def custom_multi_label_accuracy(y_true, y_pred):
        # need some threshold-specific rounding code here, presently only for 0.5 thresh.
        return K.mean(K.round(np.multiply(y_true,y_pred)),axis=0)
    
    def eval_binary_accuracy(y_test, y_pred):
        correct_indices = y_test==y_pred
        all_correct_predictions = np.zeros(y_test.shape)
        all_correct_predictions[correct_indices] = 1
        #print("DEBUG::binary accuracy matrix")
        #print(all_correct_predictions)
        return np.mean(all_correct_predictions),np.mean(all_correct_predictions, axis=0),all_correct_predictions
    
    def eval_confusion(y_test, y_pred):
        wrong_indices = y_test!=y_pred
        all_wrong_predictions = np.zeros(y_test.shape)
        all_wrong_predictions[wrong_indices] = 1
        #print("DEBUG::confusion matrix")
        #print(all_wrong_predictions)
        return np.mean(all_wrong_predictions),np.mean(all_wrong_predictions, axis=0),all_wrong_predictions
    
    def eval_false_positives(y_test, y_pred):
        false_positive_matrix = np.zeros((y_test.shape[1],y_test.shape[1]))
        false_positives = np.multiply(y_pred,1-y_test)
        # print(precision_matrix)
        for i in np.arange(y_test.shape[0]):
            for j in np.arange(y_test.shape[1]) :
                if(false_positives[i,j]==1): #positive label for ith sample and jth predicted category
                    for k in np.arange(y_test.shape[1]):
                        if(y_test[i,k]==1): #positive label for ith sample and kth true category
                            # print("DEBUG::i,j,k")
                            # print("%d,%d,%d"%(i,j,k)) 
                            false_positive_matrix[j,k] +=1
        # print("DEBUG::precision matrix")
        # print(precision_matrix)
        return np.sum(false_positive_matrix),np.sum(false_positive_matrix, axis=0),false_positive_matrix
    
    def binarize_outshape(self,in_shape):
        return in_shape[0], in_shape[1], 71

    def max_1d(x):
        return K.max(x, axis=1)

    def striphtml(html):
        p = re.compile(r'<.*?>')
        return p.sub('', html)

    def clean(s):
        return re.sub(r'[^\x00-\x7f]', r'', s)

    def setup_test_sets(self,X, y):
        ids = np.arange(len(X))
        np.random.shuffle(ids)

        # shuffle
        X = X[ids]
        y = y[ids]

        train_end = int(X.shape[0] * .6)
        cross_validation_end = int(X.shape[0] * .3 + train_end)
        test_end = int(X.shape[0] * .1 + cross_validation_end)
    
        X_train = X[:train_end]
        X_cv_test = X[train_end:cross_validation_end]
        X_test = X[cross_validation_end:test_end]

        y_train = y[:train_end]
        y_cv_test = y[train_end:cross_validation_end]
        y_test = y[cross_validation_end:test_end]

        data = type('data_type', (object,), {'X_train' : X_train, 'X_cv_test': X_cv_test, 'X_test': X_test, 'y_train': y_train, 'y_cv_test': y_cv_test, 'y_test':y_test})
        return data

    def generate_model(self,max_len, max_cells, category_count):
        filter_length = [1, 3, 3]
        nb_filter = [40, 200, 1000]
        pool_length = 2
        # document input
        document = Input(shape=(max_cells, max_len), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(max_len,), dtype='int64')
        # char indices to one hot matrix, 1D sequence to 2D
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(in_sentence)
        # embedded: encodes sentence
        for i in range(len(nb_filter)):
            embedded = Convolution1D(nb_filter=nb_filter[i],
                                     filter_length=filter_length[i],
                                     border_mode='valid',
                                     activation='relu',
                                     init='glorot_normal',
                                     subsample_length=1)(embedded)

            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_length=pool_length)(embedded)

        forward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                        dropout_U=0.2, consume_less='gpu')(embedded)
        backward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                        dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)

        sent_encode = merge([forward_sent, backward_sent],
                            mode='concat', concat_axis=-1)
        sent_encode = Dropout(0.3)(sent_encode)
        # sentence encoder

        encoder = Model(input=in_sentence, output=sent_encode)

        print(encoder.summary())
        encoded = TimeDistributed(encoder)(document)

        # encoded: sentences to bi-lstm for document encoding
        forwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                        dropout_U=0.2, consume_less='gpu')(encoded)
        backwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                        dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)

        merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
        output = Dropout(0.3)(merged)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.3)(output)
        output = Dense(category_count, activation='sigmoid')(output)
        # output = Activation('softmax')(output)
        model = Model(input=document, output=output)

        return model


    # record history of training
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accuracies = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accuracies.append(logs.get('binary_accuracy'))

    def plot_loss(history):
        import matplotlib.pyplot as plt
        # summarize history for accuracy
        plt.subplot('121')
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.subplot('122')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train_model(batch_size, checkpoint_dir, model, nb_epoch, data):
        print("starting learning")
    
        check_cb = keras.callbacks.ModelCheckpoint(checkpoint_dir + "text-class" + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                    monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

        tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)
        loss_history = LossHistory()
        history = model.fit(data.X_train, data.y_train, validation_data=(data.X_cv_test, data.y_cv_test), batch_size=batch_size,
                    nb_epoch=nb_epoch, shuffle=True, callbacks=[earlystop_cb, check_cb, loss_history, tbCallBack])
    
        print('losses: ')
        print(history.history['loss'])
        print('accuracies: ')
        # print(history.history['acc'])
        print(history.history['val_binary_accuracy'])
        #self.plot_loss(history)

    def evaluate_model(max_cells, model, data, encoder, p_threshold):
        print("Starting predictions:")
    
        start = time.time()
        scores = model.evaluate(data.X_test, data.y_test, verbose=0)
        end = time.time()
        print("Accuracy: %.2f%% \n Time: {0}s \n Time/example : {1}s/ex".format(
            end - start, (end - start) / data.X_test.shape[0]) % (scores[1] * 100))
    
        # return all predictions above a certain threshold
        # first, the maximum probability/class
        probabilities = model.predict(data.X_test, verbose=1)
        # print("The prediction probabilities are:")
        # print(probabilities)
        m = np.amax(probabilities, axis=1)
        max_index = np.argmax(probabilities, axis=1)
        # print("Associated fixed category indices:")
        # print(max_index)
        #with open('Categories.txt','r') as f:
        #        Categories = f.read().splitlines()
        Categories = encoder.categories
        print("Remember that the fixed categories are:")
        print(Categories)
        print("Most Likely Predicted Category/Labels are: ")
        print((np.array(Categories))[max_index])
        print("Associated max probabilities/confidences:")
        print(m)
        # next, all probabilities above a certain threshold
        print("DEBUG::y_test:")
        print(data.y_test)
        prediction_indices = probabilities > p_threshold
        y_pred = np.zeros(data.y_test.shape)
        y_pred[prediction_indices] = 1
        print("DEBUG::y_pred:")
        print(y_pred)
        print("'Binary' accuracy (true positives + true negatives) is:")
        print(eval_binary_accuracy(data.y_test,y_pred))
        print("'Binary' confusion (false positives + false negatives) is:")
        print(eval_confusion(data.y_test,y_pred))
        print("False positive matrix is:")
        print(eval_false_positives(data.y_test,y_pred))

    def resolve_file_path(filename, dir):
        if os.path.isfile(str(filename)):
            return str(filename)
        elif os.path.isfile(str(dir + str(filename))):
            return dir + str(filename)

    def load_weights(self,checkpoint_name, config, model, checkpoint_dir):
        if config and not checkpoint_name:
            checkpoint_name = config['checkpoint']
        if checkpoint_name:
            checkpoint_path = resolve_file_path(checkpoint_name, checkpoint_dir)
            print("Checkpoint : %s" % str(checkpoint_path))
            model.load_weights(checkpoint_path)

    def save_config(execution_config, checkpoint_dir):
        filename = ""
        if not execution_config["checkpoint"] is None:
            filename = execution_config["checkpoint"].rsplit( ".", 1 )[ 0 ] + ".pkl"
        else:
            filename = time.strftime("%Y%m%d-%H%M%S") + ".pkl"
        with open(checkpoint_dir + filename, 'wb') as output:
            pickle.dump(execution_config, output, pickle.HIGHEST_PROTOCOL)

    def load_config(execution_config_path, dir):
        execution_config_path = resolve_file_path(execution_config_path, dir)
        return pickle.load( open( execution_config_path, "rb" ) )

    def get_best_checkpoint(checkpoint_dir):
        max_mtime = 0
        max_file = ''
        for dirname,subdirs,files in os.walk(checkpoint_dir):
            for fname in files:
                full_path = os.path.join(dirname, fname)
                mtime = os.stat(full_path).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    max_dir = dirname
                    max_file = fname
        return max_file